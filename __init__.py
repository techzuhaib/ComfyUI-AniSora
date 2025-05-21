from .nodes import LoadAniSoraModel, Prompt, AniSora, SaveAniSora

NODE_CLASS_MAPPINGS = {
    "LoadAniSoraModel": LoadAniSoraModel,
    "Prompt": Prompt,
    "AniSora": AniSora,
    "SaveAniSora": SaveAniSora,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAniSoraModel": "Load AniSora Model",
    "Prompt": "Prompt",
    "AniSora": "AniSora",
    "SaveAniSora": "Save AniSora",
} 

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
