from config.settings import CONFIDENCE_THRESHOLD

def should_escalate(confidence: float) -> bool:
    return confidence < CONFIDENCE_THRESHOLD
