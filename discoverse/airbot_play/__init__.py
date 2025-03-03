try:
    from .airbot_play_fik import AirbotPlayFIK
    from .airbot_play_id import AirbotPlayID
except ImportError as e:
    from .airbot_play_ik_nopin import AirbotPlayIK_nopin

    AirbotPlayFIK = AirbotPlayIK_nopin
