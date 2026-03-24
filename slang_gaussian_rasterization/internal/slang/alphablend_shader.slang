// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//TODO: Clean-up shared-memory logic, too complicated, could be simplified.

import utils;

static const uint TILE_HEIGHT = PYTHON_TILE_HEIGHT;
static const uint TILE_WIDTH = PYTHON_TILE_WIDTH;

groupshared Splat_2D_AlphaBlend collected_splats[TILE_HEIGHT * TILE_WIDTH];
groupshared uint32_t collected_idx[TILE_HEIGHT * TILE_WIDTH];

[Differentiable]
PixelState update_pixel_state(PixelState pixel_state_t_nm1, PixelState gauss_rgba_t_n)
{
    float3 color_t_n = pixel_state_t_nm1.color.rgb + gauss_rgba_t_n.color.rgb * pixel_state_t_nm1.color.a;
    float transmittance_t_n = pixel_state_t_nm1.color.a * (1 - gauss_rgba_t_n.color.a);
    float depth_t_n = pixel_state_t_nm1.depth + gauss_rgba_t_n.depth * pixel_state_t_nm1.color.a;
    return PixelState(float4(color_t_n, transmittance_t_n), depth_t_n);
}

PixelState undo_pixel_state(PixelState pixel_state_t_n, PixelState gauss_rgba_t_n)
{
    float transmittance_t_nm1 = pixel_state_t_n.color.a / (1 - gauss_rgba_t_n.color.a);
    float3 color_t_nm1 = pixel_state_t_n.color.rgb - gauss_rgba_t_n.color.rgb * transmittance_t_nm1;
    float depth_t_nm1 = pixel_state_t_n.depth - gauss_rgba_t_n.depth * transmittance_t_nm1;
    return PixelState(float4(color_t_nm1, transmittance_t_nm1), depth_t_nm1);
}


[BackwardDerivative(bwd_alpha_blend)] // Use a custom derivative so that we can hand-write the structure of the reverse loop
PixelState alpha_blend(TensorView<int32_t> sorted_gauss_idx,
                   DiffTensorView xyz_vs,
                   DiffTensorView inv_cov_vs,
                   DiffTensorView opacity,
                   DiffTensorView rgb,
                   DiffTensorView final_pixel_state,
                   TensorView<int32_t> n_contributors,
                   uint32_t2 pix_coord,
                   uint32_t tile_idx_start,
                   uint32_t tile_idx_end,
                   uint32_t tile_height,
                   uint32_t tile_width,
                   uint32_t H,
                   uint32_t W)
{
    float2 center_pix_coord = pix_coord;
    PixelState curr_pixel_state = PixelState(float4(0.f, 0.f, 0.f, 1.f), float(0.f));
    uint32_t block_size = tile_height * tile_width;
    bool is_inside = (pix_coord.x < W && pix_coord.y < H);
    bool thread_active = is_inside;

    const int shared_memory_rounds = ((tile_idx_end - tile_idx_start + block_size - 1) / block_size);
    uint32_t thread_rank = cudaThreadIdx().y * cudaBlockDim().x + cudaThreadIdx().x;

    int32_t local_n_contrib = 0;
    int splats_left_to_process = tile_idx_end - tile_idx_start;
    for (int i = 0; i < shared_memory_rounds; i++)
    {
        // Collectively fetch per-Gaussian data from global to shared
        AllMemoryBarrierWithGroupSync();
        int splat_pointer_offset = i * block_size + thread_rank;
        if (tile_idx_start + splat_pointer_offset < tile_idx_end)
        {
            uint32_t coll_id = uint32_t(sorted_gauss_idx[tile_idx_start + splat_pointer_offset]);
            collected_splats[thread_rank] = load_splat_alphablend(coll_id, xyz_vs, inv_cov_vs, opacity, rgb);
        }
        AllMemoryBarrierWithGroupSync();
        if (thread_active) {
            for (int j = 0; j < min(block_size, splats_left_to_process); j++)
            {
                local_n_contrib++;
                Splat_2D_AlphaBlend g = collected_splats[j];
                PixelState gauss_rgba = evaluate_splat(g, center_pix_coord, H, W);
                
                // Skip Splats that have a tiny contribution.
                if (gauss_rgba.color.a < 1.0f / 255.0f)
                    continue;

                PixelState new_pixel_state = update_pixel_state(curr_pixel_state, gauss_rgba);
                if (new_pixel_state.color.a < 0.0001f) {
                    // This Splat never registred so we subtract it before we break.
                    local_n_contrib--;
                    thread_active = false;
                    break;
                }
                curr_pixel_state = new_pixel_state;
            }
        }
        splats_left_to_process -= block_size;
    }

    if (is_inside)
        n_contributors[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)] = local_n_contrib;

    return curr_pixel_state;
}

void bwd_alpha_blend(TensorView<int32_t> sorted_gauss_idx,
                     DiffTensorView xyz_vs,
                     DiffTensorView inv_cov_vs,
                     DiffTensorView opacity,
                     DiffTensorView rgb,
                     DiffTensorView final_pixel_state,
                     TensorView<int32_t> n_contributors,
                     uint32_t2 pix_coord,
                     uint32_t tile_idx_start,
                     uint32_t tile_idx_end,
                     uint32_t tile_height,
                     uint32_t tile_width,
                     uint32_t H,
                     uint32_t W,
                     PixelState d_current_pixel_state)
{
    // Load the final pixel state.
    bool is_inside = (pix_coord.x < W && pix_coord.y < H);
    uint32_t block_size = tile_height * tile_width;
    const int rounds = ((tile_idx_end - tile_idx_start + block_size - 1) / block_size);

    int splats_left_to_process = tile_idx_end - tile_idx_start;
    uint32_t current_splat_offset = tile_idx_end - tile_idx_start;

    PixelState current_pixel_state;
    int32_t n_contrib_fwd;
    if (is_inside) {
        current_pixel_state = PixelState(
            float4(final_pixel_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)],
                   final_pixel_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 1)],
                   final_pixel_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 2)],
                   final_pixel_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 3)]),
            float(final_pixel_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 4)]));
        n_contrib_fwd = n_contributors[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)];
    }

    float2 center_pix_coord = pix_coord;
    DifferentialPair<float2> dp_center_pix_coord = diffPair(center_pix_coord);

    // Damping constants (minimal bias, strong stability)
    const float EPS_SAT = 1e-3f;   // floor on (1 - a) for damping computation
    const float G_CAP   = 50.0f;   // cap on effective 1/(1-a) gain; tune 25–75 if needed

    uint32_t thread_rank = cudaThreadIdx().y * cudaBlockDim().x + cudaThreadIdx().x;
    for (int i = 0; i < rounds; i++)
    {
        // Collectively fetch per-Gaussian data from global to shared
        AllMemoryBarrierWithGroupSync();
        int progress = i * block_size + thread_rank;
        if (tile_idx_start + progress < tile_idx_end)
        {
            uint32_t coll_id = uint32_t(sorted_gauss_idx[tile_idx_end - progress - 1]);
            collected_idx[thread_rank] = coll_id;
            collected_splats[thread_rank] = load_splat_alphablend(coll_id, xyz_vs, inv_cov_vs, opacity, rgb);
        }
        AllMemoryBarrierWithGroupSync();
        if (is_inside) {
            for (int j = 0; j < min(block_size, splats_left_to_process); j++)
            {
                current_splat_offset--;
                if (current_splat_offset >= n_contrib_fwd)
                    continue;
                uint32_t g_idx = collected_idx[j];
                Splat_2D_AlphaBlend g = collected_splats[j];

                PixelState gauss_rgba = evaluate_splat(g, center_pix_coord, H, W);

                if (gauss_rgba.color.a < 1.0f / 255.0f)
                    continue;

                // Undo pixel state
                current_pixel_state = undo_pixel_state(current_pixel_state, gauss_rgba);

                // Back-prop automatically through blending and gaussian evaluation.
                DifferentialPair<Splat_2D_AlphaBlend> dp_g = diffPair(g);
                DifferentialPair<PixelState> dp_gauss_rgba = diffPair(gauss_rgba);
                DifferentialPair<PixelState> dp_current_pixel_state = diffPair(current_pixel_state);

                // ---- Minimal per-splat saturation damping (no path-length dependence) ----
                // a in [0,1). When a→1, the adjoint contains factors ~1/(1-a).
                const float a   = gauss_rgba.color.a;
                const float sat = max(1.0f - a, EPS_SAT);         // distance to saturation
                const float scale = min(1.0f, G_CAP * sat);       // cap effective gain

                PixelState d_seed = d_current_pixel_state;
                d_seed.color.a   *= scale;
                d_seed.color.rgb *= scale;
                d_seed.depth     *= scale;
                bwd_diff(update_pixel_state)(dp_current_pixel_state, dp_gauss_rgba, d_seed);
                d_current_pixel_state = dp_current_pixel_state.getDifferential();

                PixelState d_gauss = dp_gauss_rgba.d;  // make a mutable copy
                d_gauss.color.a   *= scale;
                d_gauss.color.rgb *= scale;
                d_gauss.depth     *= scale;
                bwd_diff(evaluate_splat)(dp_g, dp_center_pix_coord, H, W, d_gauss);

                // Backprop to per-splat parameters
                bwd_diff(load_splat_alphablend)(g_idx, xyz_vs, inv_cov_vs, opacity, rgb, dp_g.d);
            }
        }
        splats_left_to_process -= block_size;
    }
}

[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
void splat_tiled(TensorView<int32_t> sorted_gauss_idx,
                 TensorView<int32_t> tile_ranges,
                 DiffTensorView xyz_vs,
                 DiffTensorView inv_cov_vs,
                 DiffTensorView opacity,
                 DiffTensorView rgb,
                 DiffTensorView output_img,
                 TensorView<int32_t> n_contributors,
                 int grid_height,
                 int grid_width,
                 int tile_height,
                 int tile_width,
                 TensorView<float> bg_color)
{
    uint32_t3 globalIdx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();

    uint32_t2 pix_coord = globalIdx.xy;

    uint32_t tile_idx = cudaBlockIdx().y * grid_width + cudaBlockIdx().x;
    uint32_t tile_idx_start = uint32_t(tile_ranges[uint2(tile_idx, 0)]);
    uint32_t tile_idx_end = uint32_t(tile_ranges[uint2(tile_idx, 1)]);


    if (tile_idx_start >= tile_idx_end) {
        bool is_inside = (pix_coord.x < output_img.size(1) && pix_coord.y < output_img.size(0));
        if (is_inside) {
            output_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0), no_diff(bg_color[0]));
            output_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 1), no_diff(bg_color[1]));
            output_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 2), no_diff(bg_color[2]));
            output_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 3), 0.0f);
            output_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 4), 0.0f);
        }
        return;
    }

    bool is_inside = (pix_coord.x < output_img.size(1) && pix_coord.y < output_img.size(0));

    PixelState full_pixel_state = alpha_blend(sorted_gauss_idx,
                                     xyz_vs,
                                     inv_cov_vs,
                                     opacity,
                                     rgb,
                                     output_img,
                                     n_contributors,
                                     pix_coord,
                                     tile_idx_start,
                                     tile_idx_end,
                                     tile_height,
                                     tile_width,
                                     output_img.size(0),
                                     output_img.size(1));

    float4 pixel_state = full_pixel_state.color;

    if (is_inside) {
      output_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0),
          pixel_state.r + no_diff(bg_color[0]) * pixel_state.a);
      output_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 1),
          pixel_state.g + no_diff(bg_color[1]) * pixel_state.a);
      output_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 2),
          pixel_state.b + no_diff(bg_color[2]) * pixel_state.a);
      output_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 3), pixel_state.a);
      output_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 4), full_pixel_state.depth);
    }
}
