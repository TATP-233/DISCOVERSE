
class BaseConfig:
    mjcf_file_path = ""
    decimation     = 2
    timestep       = 0.005
    sync           = True
    headless       = False
    render_set     = {
        "fps"    : 24,
        "width"  : 1280,
        "height" :  720
    }
    obs_rgb_cam_id = None
    obs_depth_cam_id = None
    rb_link_list   = []
    obj_list       = []
    gs_model_dict  = {}
    use_gaussian_renderer = False
    enable_render = True
    max_render_depth = 5.0
    # Hugging Face 配置
    hf_repo_id = "tatp/DISCOVERSE-models"  # 默认HF仓库ID
    hf_local_dir = None  # 下载目标目录，None表示使用默认目录(DISCOVERSE_ASSETS_DIR/3dgs)