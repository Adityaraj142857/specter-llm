ROLE_LABELS = {
    "hr": "HR Department",
    "sde": "Engineering",
    "external": "External Partner",
    "legal": "Legal Team",
}

def get_role_label(role: str) -> str:
    return ROLE_LABELS.get(role, role.upper())
