from pydantic import BaseModel
from typing import Optional, Dict

class PredectionRequest(BaseModel):
    x_0: Optional[float] = None
    x_1: Optional[float] = None
    x_2: Optional[float] = None
    x_3: Optional[float] = None
    x_4: Optional[float] = None
    x_5: Optional[float] = None
    x_6: Optional[float] = None
    x_7: Optional[float] = None
    x_8: Optional[float] = None
    x_9: Optional[str] = None
    x_10: Optional[float] = None
    x_11: Optional[float] = None
    x_12: Optional[float] = None
    x_13: Optional[float] = None
    x_14: Optional[float] = None
    x_15: Optional[float] = None
    x_16: Optional[str] = None
    x_17: Optional[float] = None
    x_18: Optional[float] = None
    x_19: Optional[float] = None
    x_20: Optional[float] = None
    x_21: Optional[float] = None
    x_22: Optional[float] = None
    x_23: Optional[float] = None
    x_24: Optional[float] = None
    x_25: Optional[float] = None
    x_26: Optional[float] = None
    x_27: Optional[float] = None
    x_28: Optional[float] = None
    x_29: Optional[float] = None
    x_30: Optional[float] = None
    x_31: Optional[float] = None
    x_32: Optional[float] = None
    x_33: Optional[float] = None
    x_34: Optional[float] = None
    x_35: Optional[float] = None
    x_36: Optional[float] = None
    x_37: Optional[float] = None
    x_38: Optional[float] = None
    x_39: Optional[float] = None
    x_40: Optional[float] = None
    x_41: Optional[float] = None
    x_42: Optional[float] = None
    x_43: Optional[float] = None
    x_44: Optional[float] = None
    x_45: Optional[float] = None
    x_46: Optional[float] = None
    x_47: Optional[float] = None
    x_48: Optional[float] = None
    x_49: Optional[float] = None
    x_50: Optional[float] = None
    x_51: Optional[str] = None
    x_52: Optional[float] = None
    x_53: Optional[float] = None
    x_54: Optional[float] = None
    x_55: Optional[float] = None
    x_56: Optional[float] = None
    x_57: Optional[float] = None
    x_58: Optional[float] = None
    x_59: Optional[float] = None
    x_60: Optional[float] = None
    x_61: Optional[float] = None
    x_62: Optional[float] = None
    x_63: Optional[float] = None
    x_64: Optional[float] = None
    x_65: Optional[str] = None
    x_66: Optional[float] = None
    x_67: Optional[float] = None
    x_68: Optional[float] = None
    x_69: Optional[float] = None
    x_70: Optional[float] = None
    x_71: Optional[float] = None
    x_72: Optional[float] = None
    x_73: Optional[float] = None
    x_74: Optional[float] = None
    x_75: Optional[str] = None
    x_76: Optional[float] = None
    x_77: Optional[float] = None
    x_78: Optional[float] = None
    x_79: Optional[float] = None
    x_80: Optional[float] = None
    x_81: Optional[float] = None
    x_82: Optional[float] = None
    x_83: Optional[float] = None
    x_84: Optional[float] = None
    x_85: Optional[float] = None
    x_86: Optional[float] = None
    x_87: Optional[float] = None
    x_88: Optional[float] = None
    x_89: Optional[str] = None
    x_90: Optional[float] = None
    x_91: Optional[float] = None
    x_92: Optional[float] = None
    x_93: Optional[float] = None
    x_94: Optional[float] = None
    x_95: Optional[float] = None
    x_96: Optional[float] = None
    x_97: Optional[float] = None
    x_98: Optional[float] = None
    x_99: Optional[float] = None

class InferenceOutput(BaseModel):
    business_outcome: int
    probability: float
    feature_input: Dict[str, float] 