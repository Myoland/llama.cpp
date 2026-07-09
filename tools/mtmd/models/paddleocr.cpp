#include "models.h"

ggml_cgraph * clip_graph_paddleocr::build() {
    const int n_pos            = n_patches;
    const int num_position_ids = n_pos * 4; // m-rope requires 4 dim per position

    int mrope_sections[4] = {d_head/4, d_head/4, d_head/4, d_head/4};

    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_position_ids);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    ggml_tensor * attn_mask = nullptr;
    if (paddleocr_padded_batch) {
        attn_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_pos, n_pos, 1, n_batch);
        ggml_set_name(attn_mask, "paddleocr_attn_mask");
        ggml_set_input(attn_mask);
    }

    auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
        return ggml_rope_multi(
                    ctx0, cur, positions, nullptr,
                    d_head/2, mrope_sections, GGML_ROPE_TYPE_VISION,
                    32768, 10000, 1, 0, 1, 32, 1);
    };

    ggml_tensor * learned_pos_embd = nullptr;
    if (paddleocr_padded_batch) {
        GGML_ASSERT(batch_image_sizes.size() == (size_t) n_batch);
        const int full_width  = img.nx() / patch_size;
        const int full_height = img.ny() / patch_size;

        for (int b = 0; b < n_batch; ++b) {
            const int valid_width  = batch_image_sizes[b].width  / patch_size;
            const int valid_height = batch_image_sizes[b].height / patch_size;
            GGML_ASSERT(valid_width  > 0 && valid_width  <= full_width);
            GGML_ASSERT(valid_height > 0 && valid_height <= full_height);

            ggml_tensor * item_pos = resize_position_embeddings_to(valid_width, valid_height);
            item_pos = ggml_reshape_3d(ctx0, item_pos, n_embd, valid_width, valid_height);

            const int pad_width  = full_width  - valid_width;
            const int pad_height = full_height - valid_height;
            if (pad_width || pad_height) {
                item_pos = ggml_pad(ctx0, item_pos, 0, pad_width, pad_height, 0);
            }

            item_pos = ggml_cont_2d(ctx0, item_pos, n_embd, n_patches);
            item_pos = ggml_reshape_3d(ctx0, item_pos, n_embd, n_patches, 1);
            learned_pos_embd = learned_pos_embd == nullptr
                ? item_pos
                : ggml_concat(ctx0, learned_pos_embd, item_pos, 2);
        }
    } else {
        learned_pos_embd = resize_position_embeddings();
    }
    ggml_tensor * inp = build_inp();
    build_vit_opts vit_opts;
    vit_opts.attn_mask = attn_mask;
    ggml_tensor * cur = build_vit(
                            inp, n_patches,
                            NORM_TYPE_NORMAL,
                            hparams.ffn_op,
                            learned_pos_embd,
                            add_pos,
                            vit_opts);

    cb(cur, "vit_out", -1);

    {
        // mlp_AR paddleocr projector
        float proj_norm_eps = 1e-5;
        cur = build_norm(cur,
                    model.mm_input_norm_w, model.mm_input_norm_b,
                    NORM_TYPE_NORMAL, proj_norm_eps, -1);

        const int scale_factor = model.hparams.n_merge;
        cur = build_patch_merge_permute(cur, scale_factor);
        cur = build_ffn(cur,
                    model.mm_1_w, model.mm_1_b,
                    nullptr, nullptr,
                    model.mm_2_w, model.mm_2_b,
                    hparams.ffn_op, -1);
        cb(cur, "mlp_out", -1);
    }

    // build the graph
    ggml_build_forward_expand(gf, cur);

    return gf;
}
