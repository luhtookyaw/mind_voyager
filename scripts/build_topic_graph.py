from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT_DIR / "data" / "topic_graph"


PROMPT_TEMPLATES = {
    "emotion_exploration": [
        "What feels hardest about this for you right now?",
        "What emotions show up most strongly when this happens?",
    ],
    "thought_elicitation": [
        "What goes through your mind in that moment?",
        "What do you find yourself telling yourself when this comes up?",
    ],
    "meaning_exploration": [
        "What does that seem to mean about you or your situation?",
        "What feels most painful or important about that experience?",
    ],
    "belief_elicitation": [
        "When this happens, what do you start believing about yourself?",
        "What does this seem to say about you deep down?",
    ],
    "history_linking": [
        "Does this feeling or pattern seem familiar from earlier in your life?",
        "Have there been earlier experiences that made this feel especially sensitive?",
    ],
    "coping_exploration": [
        "How do you usually get through that moment?",
        "What do you tend to do to protect yourself when this shows up?",
    ],
    "discrepancy_elicitation": [
        "What is this pattern helping with, and what is it costing you?",
        "What part of this no longer fits with the life you want?",
    ],
    "cost_of_status_quo": [
        "What worries you most about staying in this pattern?",
        "If nothing changed here, what would concern you most?",
    ],
    "values_elicitation": [
        "What kind of person do you want to be in moments like this?",
        "What matters to you that this problem gets in the way of?",
    ],
    "strengths_exceptions": [
        "Have there been moments when this felt even a little easier to handle?",
        "What has helped you resist or step out of this pattern before?",
    ],
    "future_projection": [
        "If things improved even a little, what would look different?",
        "What would you hope to be doing more of if this felt less powerful?",
    ],
    "action_readiness": [
        "What feels like one small step you might be willing to try?",
        "What feels most realistic to experiment with before we talk again?",
    ],
}


SUPER_TOPICS = {
    "family": {
        "label": "Family",
        "description": "Family relationships, attachment experiences, roles, and conflict.",
        "aliases": ["family", "parents", "home", "caregivers", "relatives"],
        "subtopics": [
            "family_conflict",
            "family_rejection",
            "family_neglect",
            "family_criticism",
            "family_distance",
            "family_pressure",
            "caregiver_absence",
        ],
    },
    "relationships": {
        "label": "Relationships",
        "description": "Romantic and interpersonal closeness, trust, rejection, and intimacy.",
        "aliases": ["relationship", "partner", "dating", "friendship", "closeness"],
        "subtopics": [
            "relationship_conflict",
            "relationship_rejection",
            "abandonment_fears",
            "betrayal",
            "intimacy_fear",
            "social_disconnection",
        ],
    },
    "health": {
        "label": "Health",
        "description": "Physical health, body-related concerns, illness, and medical stress.",
        "aliases": ["health", "medical", "body", "illness", "physical"],
        "subtopics": [
            "medical_illness",
            "body_image",
            "chronic_condition",
            "sleep_disturbance",
            "pain_fatigue",
        ],
    },
    "mental_health": {
        "label": "Mental Health",
        "description": "Psychological symptoms, emotional regulation, therapy, and psychiatric treatment.",
        "aliases": ["mental health", "depression", "anxiety", "stress", "therapy", "psychiatric"],
        "subtopics": [
            "depression_symptoms",
            "anxiety_symptoms",
            "stress_overload",
            "trauma_response",
            "emotion_dysregulation",
            "therapy_history",
            "psychiatric_medication",
        ],
    },
    "education": {
        "label": "Education",
        "description": "School-based stress, academic experiences, peer dynamics, and exclusion.",
        "aliases": ["school", "education", "college", "class", "academic"],
        "subtopics": [
            "school_bullying",
            "academic_failure",
            "peer_exclusion",
            "performance_pressure",
        ],
    },
    "work": {
        "label": "Work",
        "description": "Employment, performance, role demands, and occupational stress.",
        "aliases": ["work", "job", "career", "boss", "workplace"],
        "subtopics": [
            "job_loss",
            "work_stress",
            "burnout",
            "workplace_conflict",
            "role_insecurity",
        ],
    },
    "economy": {
        "label": "Economy",
        "description": "Financial hardship, dependency, debt, compensation, and treatment-related costs.",
        "aliases": ["economy", "money", "finance", "debt", "bills", "insurance", "compensation"],
        "subtopics": [
            "financial_hardship",
            "debt_pressure",
            "financial_dependency",
            "insurance_costs",
            "compensation_support",
        ],
    },
    "law": {
        "label": "Law",
        "description": "Legal pressure, custody conflict, and fear of institutions or punishment.",
        "aliases": ["law", "court", "legal", "custody", "police"],
        "subtopics": [
            "legal_trouble",
            "custody_conflict",
            "institutional_fear",
        ],
    },
    "identity": {
        "label": "Identity",
        "description": "Self-worth, sexuality, belonging, competence, independence, values, and growth.",
        "aliases": ["identity", "self-worth", "sexuality", "belonging", "competence", "values"],
        "subtopics": [
            "identity_shame",
            "worthlessness_theme",
            "defectiveness_theme",
            "not_belonging",
            "failure_identity_theme",
            "sexuality_identity",
            "independence_theme",
            "competence_theme",
            "personal_growth_theme",
        ],
    },
    "social_life": {
        "label": "Social Life",
        "description": "Friendships, groups, community, roommates, peer rejection, and social isolation.",
        "aliases": ["social life", "friendships", "community", "roommates", "social groups"],
        "subtopics": [
            "friendship_strain",
            "peer_rejection",
            "community_disconnection",
            "roommate_stress",
            "social_isolation",
            "caretaking_roles",
        ],
    },
    "housing_home": {
        "label": "Housing/Home",
        "description": "Living situation, moving, domestic stress, household duties, and home stability.",
        "aliases": ["housing", "home", "living situation", "moving", "household", "domestic"],
        "subtopics": [
            "housing_instability",
            "moving_stress",
            "household_duties",
            "domestic_stress",
            "living_arrangement_tension",
            "home_instability",
        ],
    },
    "loss": {
        "label": "Loss",
        "description": "Bereavement, separation, rupture, and meaningful losses.",
        "aliases": ["loss", "grief", "bereavement", "separation", "mourning"],
        "subtopics": [
            "grief_loss",
            "bereavement",
            "separation_loss",
            "missed_goodbye",
            "role_status_loss",
        ],
    },
    "trauma": {
        "label": "Trauma",
        "description": "Victimization, unsafe experiences, humiliation, and trauma-related vulnerability.",
        "aliases": ["trauma", "unsafe", "abuse", "victimization", "assault"],
        "subtopics": [
            "trauma_victimization",
            "unsafe_environment",
            "humiliation",
            "threat_exposure",
        ],
    },
    "coping": {
        "label": "Coping",
        "description": "Patterns used to manage distress, control exposure, or regulate feelings.",
        "aliases": ["coping", "avoid", "manage", "protect", "deal with"],
        "subtopics": [
            "avoidance_pattern",
            "withdrawal_pattern",
            "perfectionism_pattern",
            "rumination_pattern",
            "self_attack_pattern",
            "overcontrol_pattern",
        ],
    },
    "addiction": {
        "label": "Addiction",
        "description": "Substance use, relapse risk, craving, and escape behavior.",
        "aliases": ["addiction", "substance", "using", "relapse", "craving"],
        "subtopics": [
            "substance_use",
            "relapse_risk",
            "craving_escape",
        ],
    },
    "safety": {
        "label": "Safety",
        "description": "Threat sensitivity, hypervigilance, and fear of danger or collapse.",
        "aliases": ["safety", "danger", "threat", "fear", "security"],
        "subtopics": [
            "safety_hypervigilance",
            "control_loss_theme",
            "loneliness_exposure",
        ],
    },
}


SUBTOPICS = {
    "family_conflict": {"label": "Family conflict", "aliases": ["arguments", "conflict at home"]},
    "family_rejection": {"label": "Family rejection", "aliases": ["rejected by family", "not accepted"]},
    "family_neglect": {"label": "Family neglect", "aliases": ["emotionally neglected", "ignored at home"]},
    "family_criticism": {"label": "Family criticism", "aliases": ["criticized at home", "judged by family"]},
    "family_distance": {"label": "Family distance", "aliases": ["distant family", "cut off from family"]},
    "family_pressure": {"label": "Family pressure", "aliases": ["pressure from family", "expectations at home"]},
    "caregiver_absence": {"label": "Caregiver absence", "aliases": ["absent parent", "caregiver unavailable"]},
    "relationship_conflict": {"label": "Relationship conflict", "aliases": ["partner conflict", "interpersonal conflict"]},
    "relationship_rejection": {"label": "Relationship rejection", "aliases": ["rejected by partner", "romantic rejection"]},
    "abandonment_fears": {"label": "Abandonment fears", "aliases": ["fear of abandonment", "left alone"]},
    "betrayal": {"label": "Betrayal", "aliases": ["betrayed", "trust broken"]},
    "intimacy_fear": {"label": "Fear of intimacy", "aliases": ["afraid of closeness", "fear of vulnerability"]},
    "social_disconnection": {"label": "Social disconnection", "aliases": ["social isolation", "disconnected from others"]},
    "medical_illness": {"label": "Medical illness", "aliases": ["illness", "medical condition"]},
    "body_image": {"label": "Body image distress", "aliases": ["appearance shame", "body shame"]},
    "chronic_condition": {"label": "Chronic condition", "aliases": ["chronic illness", "long-term condition"]},
    "sleep_disturbance": {"label": "Sleep disturbance", "aliases": ["sleep problems", "insomnia"]},
    "pain_fatigue": {"label": "Pain or fatigue", "aliases": ["pain", "fatigue", "exhaustion"]},
    "depression_symptoms": {"label": "Depression symptoms", "aliases": ["depression", "depressed mood", "low mood"]},
    "anxiety_symptoms": {"label": "Anxiety symptoms", "aliases": ["anxiety", "panic", "worry"]},
    "stress_overload": {"label": "Stress overload", "aliases": ["stress", "overwhelmed", "pressure"]},
    "trauma_response": {"label": "Trauma response", "aliases": ["trauma response", "triggered", "flashbacks"]},
    "emotion_dysregulation": {"label": "Emotional dysregulation", "aliases": ["emotion regulation problems", "emotional swings"]},
    "therapy_history": {"label": "Therapy history", "aliases": ["therapy", "counseling", "treatment history"]},
    "psychiatric_medication": {"label": "Psychiatric medication", "aliases": ["psychiatric medication", "antidepressants", "medication for mental health"]},
    "school_bullying": {"label": "School bullying", "aliases": ["bullied at school", "peer bullying"]},
    "academic_failure": {"label": "Academic failure", "aliases": ["failed school", "academic struggles"]},
    "peer_exclusion": {"label": "Peer exclusion", "aliases": ["left out", "excluded by peers"]},
    "performance_pressure": {"label": "Performance pressure", "aliases": ["pressure to perform", "academic pressure"]},
    "job_loss": {"label": "Job loss", "aliases": ["lost my job", "unemployment"]},
    "work_stress": {"label": "Work stress", "aliases": ["job stress", "work pressure"]},
    "burnout": {"label": "Burnout", "aliases": ["burned out", "depleted by work"]},
    "workplace_conflict": {"label": "Workplace conflict", "aliases": ["conflict at work", "trouble with coworkers"]},
    "role_insecurity": {"label": "Role insecurity", "aliases": ["job insecurity", "not good enough at work"]},
    "financial_hardship": {"label": "Financial hardship", "aliases": ["money problems", "financial stress"]},
    "debt_pressure": {"label": "Debt pressure", "aliases": ["debt", "owing money"]},
    "financial_dependency": {"label": "Financial dependency", "aliases": ["dependent on others financially"]},
    "insurance_costs": {"label": "Insurance and treatment costs", "aliases": ["insurance problems", "cost of treatment"]},
    "compensation_support": {"label": "Compensation or support disputes", "aliases": ["compensation", "child support", "financial support conflict"]},
    "housing_instability": {"label": "Housing instability", "aliases": ["unstable housing", "risk of losing housing"]},
    "legal_trouble": {"label": "Legal trouble", "aliases": ["legal issues", "trouble with the law"]},
    "custody_conflict": {"label": "Custody conflict", "aliases": ["custody issues", "custody fight"]},
    "institutional_fear": {"label": "Institutional fear", "aliases": ["fear of authorities", "fear of systems"]},
    "sexuality_identity": {"label": "Sexuality and identity stress", "aliases": ["sexuality", "sexual identity", "identity conflict"]},
    "independence_theme": {"label": "Independence struggle", "aliases": ["independence", "dependence", "self-sufficiency"]},
    "competence_theme": {"label": "Competence concern", "aliases": ["competence", "capability", "not competent"]},
    "personal_growth_theme": {"label": "Personal growth and values", "aliases": ["personal growth", "values", "who I want to be"]},
    "friendship_strain": {"label": "Friendship strain", "aliases": ["friendship issues", "friend conflict"]},
    "peer_rejection": {"label": "Peer rejection", "aliases": ["peer rejection", "rejected by peers"]},
    "community_disconnection": {"label": "Community disconnection", "aliases": ["disconnected from community", "not part of a group"]},
    "roommate_stress": {"label": "Roommate stress", "aliases": ["roommate conflict", "stress with roommates"]},
    "social_isolation": {"label": "Social isolation", "aliases": ["social isolation", "isolated socially"]},
    "caretaking_roles": {"label": "Caretaking role burden", "aliases": ["caretaking role", "taking care of others"]},
    "moving_stress": {"label": "Moving stress", "aliases": ["moving", "relocation stress"]},
    "household_duties": {"label": "Household duty burden", "aliases": ["household duties", "domestic chores"]},
    "domestic_stress": {"label": "Domestic stress", "aliases": ["home stress", "domestic stress"]},
    "living_arrangement_tension": {"label": "Living arrangement tension", "aliases": ["living with parents", "living with partner", "living with roommates"]},
    "home_instability": {"label": "Domestic instability", "aliases": ["unstable home", "domestic instability"]},
    "grief_loss": {"label": "Significant loss", "aliases": ["loss", "grief"]},
    "bereavement": {"label": "Bereavement", "aliases": ["death loss", "mourning"]},
    "separation_loss": {"label": "Separation loss", "aliases": ["breakup", "separation"]},
    "missed_goodbye": {"label": "Missed goodbye", "aliases": ["missed goodbye", "could not say goodbye"]},
    "role_status_loss": {"label": "Loss of role or status", "aliases": ["loss of role", "loss of status"]},
    "trauma_victimization": {"label": "Victimization", "aliases": ["victimized", "abused", "targeted"]},
    "unsafe_environment": {"label": "Unsafe environment", "aliases": ["unsafe home", "unsafe setting"]},
    "humiliation": {"label": "Humiliation", "aliases": ["humiliated", "ashamed in front of others"]},
    "threat_exposure": {"label": "Threat exposure", "aliases": ["threatened", "felt in danger"]},
    "identity_shame": {"label": "Identity shame", "aliases": ["ashamed of who I am"]},
    "worthlessness_theme": {"label": "Worthlessness theme", "aliases": ["worthless", "not valuable"]},
    "defectiveness_theme": {"label": "Defectiveness theme", "aliases": ["defective", "broken"]},
    "not_belonging": {"label": "Not belonging", "aliases": ["do not fit in", "do not belong"]},
    "failure_identity_theme": {"label": "Failure identity", "aliases": ["I am a failure", "loser"]},
    "avoidance_pattern": {"label": "Avoidance pattern", "aliases": ["avoidance", "staying away"]},
    "withdrawal_pattern": {"label": "Withdrawal pattern", "aliases": ["withdraw", "pull away"]},
    "perfectionism_pattern": {"label": "Perfectionism pattern", "aliases": ["must get it right", "perfectionistic"]},
    "rumination_pattern": {"label": "Rumination pattern", "aliases": ["overthinking", "rumination"]},
    "self_attack_pattern": {"label": "Self-attack pattern", "aliases": ["beating myself up", "self-criticism"]},
    "overcontrol_pattern": {"label": "Overcontrol pattern", "aliases": ["must stay in control", "overcontrol"]},
    "substance_use": {"label": "Substance use", "aliases": ["using", "drinking", "drug use"]},
    "relapse_risk": {"label": "Relapse risk", "aliases": ["relapse", "using again"]},
    "craving_escape": {"label": "Craving escape", "aliases": ["urge to escape", "craving relief"]},
    "safety_hypervigilance": {"label": "Hypervigilance", "aliases": ["on guard", "hypervigilant"]},
    "control_loss_theme": {"label": "Loss of control", "aliases": ["out of control", "cannot control myself"]},
    "loneliness_exposure": {"label": "Loneliness exposure", "aliases": ["alone", "left alone", "isolated"]},
}


BELIEF_CLUSTERS = {
    "helplessness": "Sense of weakness, incompetence, entrapment, or inability to cope.",
    "unlovability": "Sense of being rejected, unwanted, alone, or impossible to love.",
    "worthlessness": "Sense of having no value, goodness, or right to matter.",
    "defectiveness": "Sense of being damaged, broken, toxic, or fundamentally flawed.",
    "danger": "Sense that the world or other people are unsafe or threatening.",
    "mistrust": "Expectation that others will exploit, betray, or harm.",
    "abandonment": "Expectation of being left, dropped, or emotionally deserted.",
    "failure_identity": "Identity-level conviction of being a failure or loser.",
    "shame_identity": "Identity-level experience of being exposed as inadequate or bad.",
    "loss_of_control": "Belief that urges, emotions, or behavior cannot be contained.",
}


INTERMEDIATE_TYPES = {
    "rejection_rule": "Conditional belief that authenticity or neediness leads to rejection.",
    "avoidance_rule": "Rule that avoidance is necessary to stay safe or cope.",
    "perfectionism_rule": "Rule that mistakes or imperfection are intolerable.",
    "control_rule": "Rule that strict control is necessary to prevent collapse or harm.",
    "self_silencing_rule": "Rule to hide needs, feelings, or identity to protect attachment.",
    "mistrust_rule": "Assumption that closeness leads to harm or betrayal.",
    "hopelessness_rule": "Assumption that change is futile or effort will fail.",
    "dependency_rule": "Assumption that one cannot cope alone and must depend on others.",
    "overresponsibility_rule": "Assumption that one must manage others or prevent bad outcomes.",
    "emotional_inhibition_rule": "Assumption that emotions should be hidden or shut down.",
}


COPING_TYPES = {
    "avoidance": "Staying away from people, places, feelings, or tasks.",
    "withdrawal": "Pulling back from contact, support, or engagement.",
    "escape": "Using distraction, fantasy, or numbing to leave distress.",
    "suppression": "Pushing thoughts, needs, or feelings down.",
    "people_pleasing": "Appeasing or adapting to maintain acceptance or reduce conflict.",
    "reassurance_seeking": "Seeking external confirmation to regulate insecurity.",
    "overplanning": "Relying on planning or preparation to reduce uncertainty.",
    "self_attack": "Using harsh self-criticism as a control or motivational strategy.",
    "substance_use_coping": "Using substances to alter mood, avoid pain, or escape.",
    "emotional_control": "Trying to tightly regulate affect or vulnerability.",
    "safety_behaviors": "Using protective behaviors to reduce perceived threat.",
    "rumination": "Repetitive mental reviewing, replaying, or analysis.",
    "help_seeking": "Turning toward others for support or guidance.",
    "activity_scheduling": "Using structured pleasant or mastery activities adaptively.",
}


EMOTION_CLUSTERS = {
    "anxiety": "Fear, worry, dread, or threat sensitivity.",
    "fear": "Immediate threat, alarm, or acute vulnerability.",
    "shame": "Humiliation, exposure, self-conscious pain, or defectiveness.",
    "sadness": "Loss, grief, low mood, or discouragement.",
    "anger": "Frustration, resentment, irritation, or rage.",
    "irritability": "Low frustration tolerance, agitation, or touchiness.",
    "guilt": "Self-blame, remorse, or moral distress.",
    "loneliness": "Isolation, disconnection, or unmet attachment need.",
    "hopelessness": "Defeat, futility, or no-way-out despair.",
    "numbness": "Disconnection, shutdown, or emotional absence.",
}


BEHAVIOR_PATTERNS = {
    "social_avoidance": "Avoiding exposure, gatherings, or interpersonal risk.",
    "conflict_avoidance": "Avoiding confrontation, disagreement, or emotionally charged exchange.",
    "isolation": "Reducing contact and closing off from support or closeness.",
    "self_attacking_behavior": "Behaviorally reinforcing self-criticism, punishment, or harsh self-treatment.",
    "overworking": "Working excessively to escape distress or prove worth.",
    "procrastination": "Delaying because of fear, shame, or overwhelm.",
    "checking_monitoring": "Monitoring for danger, mistakes, or reassurance.",
    "appeasement": "Complying or softening to reduce risk of conflict or rejection.",
    "emotional_withdrawal": "Pulling back affectively while staying physically present.",
    "relapse_behavior": "Returning to addictive or high-risk coping patterns.",
    "routine_control": "Using rigid routines or structure to create safety.",
    "reassurance_loop": "Repeatedly seeking confirmation or checking closeness.",
}


SUBTOPIC_MAPS = {
    "family_conflict": {"beliefs": ["mistrust", "danger"], "intermediates": ["avoidance_rule", "overresponsibility_rule"], "coping": ["avoidance", "people_pleasing"], "emotions": ["anxiety", "anger"], "behaviors": ["conflict_avoidance", "appeasement"], "prompts": ["emotion_exploration", "coping_exploration", "discrepancy_elicitation"]},
    "family_rejection": {"beliefs": ["unlovability", "abandonment"], "intermediates": ["rejection_rule", "self_silencing_rule"], "coping": ["withdrawal", "people_pleasing"], "emotions": ["shame", "loneliness"], "behaviors": ["isolation", "appeasement"], "prompts": ["belief_elicitation", "history_linking", "values_elicitation"]},
    "family_neglect": {"beliefs": ["unlovability", "worthlessness"], "intermediates": ["hopelessness_rule", "self_silencing_rule"], "coping": ["withdrawal", "suppression"], "emotions": ["sadness", "loneliness"], "behaviors": ["emotional_withdrawal", "isolation"], "prompts": ["history_linking", "emotion_exploration", "belief_elicitation"]},
    "family_criticism": {"beliefs": ["defectiveness", "failure_identity"], "intermediates": ["perfectionism_rule", "self_silencing_rule"], "coping": ["self_attack", "people_pleasing"], "emotions": ["shame", "anxiety"], "behaviors": ["appeasement", "overworking"], "prompts": ["meaning_exploration", "belief_elicitation", "cost_of_status_quo"]},
    "family_distance": {"beliefs": ["abandonment", "unlovability"], "intermediates": ["mistrust_rule", "rejection_rule"], "coping": ["withdrawal", "suppression"], "emotions": ["loneliness", "sadness"], "behaviors": ["isolation", "emotional_withdrawal"], "prompts": ["emotion_exploration", "history_linking", "future_projection"]},
    "family_pressure": {"beliefs": ["failure_identity", "worthlessness"], "intermediates": ["perfectionism_rule", "overresponsibility_rule"], "coping": ["overplanning", "self_attack"], "emotions": ["anxiety", "shame"], "behaviors": ["overworking", "procrastination"], "prompts": ["thought_elicitation", "belief_elicitation", "discrepancy_elicitation"]},
    "caregiver_absence": {"beliefs": ["abandonment", "unlovability"], "intermediates": ["dependency_rule", "self_silencing_rule"], "coping": ["withdrawal", "people_pleasing"], "emotions": ["loneliness", "sadness"], "behaviors": ["isolation", "reassurance_loop"], "prompts": ["history_linking", "emotion_exploration", "strengths_exceptions"]},
    "relationship_conflict": {"beliefs": ["mistrust", "danger"], "intermediates": ["avoidance_rule", "mistrust_rule"], "coping": ["avoidance", "people_pleasing"], "emotions": ["anger", "anxiety"], "behaviors": ["conflict_avoidance", "appeasement"], "prompts": ["emotion_exploration", "coping_exploration", "values_elicitation"]},
    "relationship_rejection": {"beliefs": ["unlovability", "defectiveness"], "intermediates": ["rejection_rule", "self_silencing_rule"], "coping": ["withdrawal", "suppression"], "emotions": ["shame", "sadness"], "behaviors": ["isolation", "emotional_withdrawal"], "prompts": ["belief_elicitation", "meaning_exploration", "future_projection"]},
    "abandonment_fears": {"beliefs": ["abandonment", "unlovability"], "intermediates": ["dependency_rule", "rejection_rule"], "coping": ["reassurance_seeking", "people_pleasing"], "emotions": ["anxiety", "loneliness"], "behaviors": ["reassurance_loop", "appeasement"], "prompts": ["thought_elicitation", "coping_exploration", "strengths_exceptions"]},
    "betrayal": {"beliefs": ["mistrust", "danger"], "intermediates": ["mistrust_rule", "control_rule"], "coping": ["withdrawal", "safety_behaviors"], "emotions": ["anger", "anxiety"], "behaviors": ["emotional_withdrawal", "checking_monitoring"], "prompts": ["history_linking", "meaning_exploration", "values_elicitation"]},
    "intimacy_fear": {"beliefs": ["unlovability", "danger"], "intermediates": ["rejection_rule", "emotional_inhibition_rule"], "coping": ["suppression", "avoidance"], "emotions": ["anxiety", "shame"], "behaviors": ["emotional_withdrawal", "social_avoidance"], "prompts": ["belief_elicitation", "emotion_exploration", "future_projection"]},
    "social_disconnection": {"beliefs": ["unlovability", "worthlessness"], "intermediates": ["rejection_rule", "avoidance_rule"], "coping": ["withdrawal", "rumination"], "emotions": ["loneliness", "shame"], "behaviors": ["isolation", "social_avoidance"], "prompts": ["emotion_exploration", "coping_exploration", "future_projection"]},
    "medical_illness": {"beliefs": ["helplessness", "danger"], "intermediates": ["avoidance_rule", "dependency_rule"], "coping": ["help_seeking", "overplanning"], "emotions": ["anxiety", "sadness"], "behaviors": ["routine_control", "checking_monitoring"], "prompts": ["meaning_exploration", "coping_exploration", "action_readiness"]},
    "body_image": {"beliefs": ["defectiveness", "unlovability"], "intermediates": ["rejection_rule", "perfectionism_rule"], "coping": ["avoidance", "self_attack"], "emotions": ["shame", "anxiety"], "behaviors": ["social_avoidance", "self_attacking_behavior"], "prompts": ["belief_elicitation", "emotion_exploration", "values_elicitation"]},
    "chronic_condition": {"beliefs": ["helplessness", "loss_of_control"], "intermediates": ["hopelessness_rule", "control_rule"], "coping": ["overplanning", "suppression"], "emotions": ["sadness", "hopelessness"], "behaviors": ["routine_control", "emotional_withdrawal"], "prompts": ["meaning_exploration", "cost_of_status_quo", "strengths_exceptions"]},
    "sleep_disturbance": {"beliefs": ["danger", "loss_of_control"], "intermediates": ["control_rule", "hopelessness_rule"], "coping": ["rumination", "overplanning"], "emotions": ["anxiety", "hopelessness"], "behaviors": ["checking_monitoring", "routine_control"], "prompts": ["thought_elicitation", "coping_exploration", "action_readiness"]},
    "pain_fatigue": {"beliefs": ["helplessness", "worthlessness"], "intermediates": ["hopelessness_rule", "avoidance_rule"], "coping": ["withdrawal", "suppression"], "emotions": ["sadness", "irritability"], "behaviors": ["isolation", "procrastination"], "prompts": ["emotion_exploration", "meaning_exploration", "strengths_exceptions"]},
    "school_bullying": {"beliefs": ["defectiveness", "unlovability"], "intermediates": ["rejection_rule", "avoidance_rule"], "coping": ["withdrawal", "self_attack"], "emotions": ["shame", "anger"], "behaviors": ["social_avoidance", "isolation"], "prompts": ["history_linking", "belief_elicitation", "future_projection"]},
    "academic_failure": {"beliefs": ["failure_identity", "worthlessness"], "intermediates": ["hopelessness_rule", "perfectionism_rule"], "coping": ["avoidance", "self_attack"], "emotions": ["shame", "hopelessness"], "behaviors": ["procrastination", "emotional_withdrawal"], "prompts": ["meaning_exploration", "belief_elicitation", "strengths_exceptions"]},
    "peer_exclusion": {"beliefs": ["unlovability", "worthlessness"], "intermediates": ["rejection_rule", "self_silencing_rule"], "coping": ["withdrawal", "people_pleasing"], "emotions": ["loneliness", "shame"], "behaviors": ["isolation", "appeasement"], "prompts": ["emotion_exploration", "history_linking", "values_elicitation"]},
    "performance_pressure": {"beliefs": ["failure_identity", "worthlessness"], "intermediates": ["perfectionism_rule", "control_rule"], "coping": ["overplanning", "self_attack"], "emotions": ["anxiety", "shame"], "behaviors": ["overworking", "procrastination"], "prompts": ["thought_elicitation", "discrepancy_elicitation", "action_readiness"]},
    "job_loss": {"beliefs": ["failure_identity", "worthlessness"], "intermediates": ["hopelessness_rule", "dependency_rule"], "coping": ["withdrawal", "rumination"], "emotions": ["shame", "hopelessness"], "behaviors": ["isolation", "procrastination"], "prompts": ["meaning_exploration", "cost_of_status_quo", "future_projection"]},
    "work_stress": {"beliefs": ["helplessness", "danger"], "intermediates": ["control_rule", "perfectionism_rule"], "coping": ["overplanning", "suppression"], "emotions": ["anxiety", "anger"], "behaviors": ["overworking", "checking_monitoring"], "prompts": ["emotion_exploration", "coping_exploration", "discrepancy_elicitation"]},
    "burnout": {"beliefs": ["helplessness", "worthlessness"], "intermediates": ["overresponsibility_rule", "hopelessness_rule"], "coping": ["withdrawal", "escape"], "emotions": ["numbness", "hopelessness"], "behaviors": ["isolation", "emotional_withdrawal"], "prompts": ["cost_of_status_quo", "values_elicitation", "action_readiness"]},
    "workplace_conflict": {"beliefs": ["mistrust", "failure_identity"], "intermediates": ["mistrust_rule", "avoidance_rule"], "coping": ["avoidance", "people_pleasing"], "emotions": ["anger", "anxiety"], "behaviors": ["conflict_avoidance", "appeasement"], "prompts": ["thought_elicitation", "coping_exploration", "values_elicitation"]},
    "role_insecurity": {"beliefs": ["failure_identity", "defectiveness"], "intermediates": ["perfectionism_rule", "control_rule"], "coping": ["overplanning", "self_attack"], "emotions": ["anxiety", "shame"], "behaviors": ["checking_monitoring", "overworking"], "prompts": ["belief_elicitation", "discrepancy_elicitation", "future_projection"]},
    "financial_hardship": {"beliefs": ["helplessness", "danger"], "intermediates": ["hopelessness_rule", "overresponsibility_rule"], "coping": ["avoidance", "overplanning"], "emotions": ["anxiety", "hopelessness"], "behaviors": ["procrastination", "checking_monitoring"], "prompts": ["cost_of_status_quo", "coping_exploration", "action_readiness"]},
    "debt_pressure": {"beliefs": ["helplessness", "worthlessness"], "intermediates": ["hopelessness_rule", "control_rule"], "coping": ["avoidance", "rumination"], "emotions": ["anxiety", "shame"], "behaviors": ["checking_monitoring", "emotional_withdrawal"], "prompts": ["thought_elicitation", "cost_of_status_quo", "action_readiness"]},
    "financial_dependency": {"beliefs": ["helplessness", "shame_identity"], "intermediates": ["dependency_rule", "self_silencing_rule"], "coping": ["people_pleasing", "suppression"], "emotions": ["shame", "anxiety"], "behaviors": ["appeasement", "emotional_withdrawal"], "prompts": ["meaning_exploration", "values_elicitation", "future_projection"]},
    "housing_instability": {"beliefs": ["danger", "helplessness"], "intermediates": ["control_rule", "hopelessness_rule"], "coping": ["overplanning", "help_seeking"], "emotions": ["anxiety", "hopelessness"], "behaviors": ["checking_monitoring", "routine_control"], "prompts": ["emotion_exploration", "coping_exploration", "action_readiness"]},
    "legal_trouble": {"beliefs": ["worthlessness", "danger"], "intermediates": ["hopelessness_rule", "mistrust_rule"], "coping": ["avoidance", "suppression"], "emotions": ["fear", "shame"], "behaviors": ["emotional_withdrawal", "checking_monitoring"], "prompts": ["meaning_exploration", "cost_of_status_quo", "future_projection"]},
    "custody_conflict": {"beliefs": ["abandonment", "danger"], "intermediates": ["overresponsibility_rule", "mistrust_rule"], "coping": ["people_pleasing", "control_rule"], "emotions": ["anxiety", "anger"], "behaviors": ["appeasement", "checking_monitoring"], "prompts": ["emotion_exploration", "values_elicitation", "action_readiness"]},
    "institutional_fear": {"beliefs": ["danger", "mistrust"], "intermediates": ["mistrust_rule", "avoidance_rule"], "coping": ["avoidance", "safety_behaviors"], "emotions": ["anxiety", "anger"], "behaviors": ["checking_monitoring", "social_avoidance"], "prompts": ["thought_elicitation", "coping_exploration", "future_projection"]},
    "grief_loss": {"beliefs": ["abandonment", "helplessness"], "intermediates": ["hopelessness_rule", "emotional_inhibition_rule"], "coping": ["withdrawal", "suppression"], "emotions": ["sadness", "loneliness"], "behaviors": ["isolation", "emotional_withdrawal"], "prompts": ["emotion_exploration", "history_linking", "future_projection"]},
    "bereavement": {"beliefs": ["abandonment", "worthlessness"], "intermediates": ["hopelessness_rule", "emotional_inhibition_rule"], "coping": ["withdrawal", "help_seeking"], "emotions": ["sadness", "numbness"], "behaviors": ["isolation", "emotional_withdrawal"], "prompts": ["emotion_exploration", "strengths_exceptions", "future_projection"]},
    "separation_loss": {"beliefs": ["abandonment", "unlovability"], "intermediates": ["rejection_rule", "hopelessness_rule"], "coping": ["withdrawal", "reassurance_seeking"], "emotions": ["sadness", "anxiety"], "behaviors": ["reassurance_loop", "isolation"], "prompts": ["belief_elicitation", "coping_exploration", "values_elicitation"]},
    "trauma_victimization": {"beliefs": ["danger", "defectiveness"], "intermediates": ["mistrust_rule", "avoidance_rule"], "coping": ["withdrawal", "safety_behaviors"], "emotions": ["fear", "shame"], "behaviors": ["social_avoidance", "checking_monitoring"], "prompts": ["emotion_exploration", "coping_exploration", "history_linking"]},
    "unsafe_environment": {"beliefs": ["danger", "helplessness"], "intermediates": ["control_rule", "avoidance_rule"], "coping": ["safety_behaviors", "overplanning"], "emotions": ["anxiety", "anger"], "behaviors": ["checking_monitoring", "routine_control"], "prompts": ["thought_elicitation", "coping_exploration", "strengths_exceptions"]},
    "humiliation": {"beliefs": ["shame_identity", "defectiveness"], "intermediates": ["self_silencing_rule", "avoidance_rule"], "coping": ["withdrawal", "self_attack"], "emotions": ["shame", "anger"], "behaviors": ["isolation", "emotional_withdrawal"], "prompts": ["meaning_exploration", "belief_elicitation", "future_projection"]},
    "threat_exposure": {"beliefs": ["danger", "mistrust"], "intermediates": ["mistrust_rule", "control_rule"], "coping": ["safety_behaviors", "suppression"], "emotions": ["anxiety", "numbness"], "behaviors": ["checking_monitoring", "emotional_withdrawal"], "prompts": ["emotion_exploration", "coping_exploration", "action_readiness"]},
    "identity_shame": {"beliefs": ["shame_identity", "defectiveness"], "intermediates": ["self_silencing_rule", "rejection_rule"], "coping": ["suppression", "withdrawal"], "emotions": ["shame", "loneliness"], "behaviors": ["emotional_withdrawal", "social_avoidance"], "prompts": ["belief_elicitation", "meaning_exploration", "values_elicitation"]},
    "worthlessness_theme": {"beliefs": ["worthlessness"], "intermediates": ["hopelessness_rule", "self_silencing_rule"], "coping": ["self_attack", "withdrawal"], "emotions": ["shame", "hopelessness"], "behaviors": ["isolation", "procrastination"], "prompts": ["belief_elicitation", "cost_of_status_quo", "strengths_exceptions"]},
    "defectiveness_theme": {"beliefs": ["defectiveness", "shame_identity"], "intermediates": ["rejection_rule", "self_silencing_rule"], "coping": ["suppression", "self_attack"], "emotions": ["shame", "anxiety"], "behaviors": ["social_avoidance", "emotional_withdrawal"], "prompts": ["belief_elicitation", "meaning_exploration", "future_projection"]},
    "not_belonging": {"beliefs": ["unlovability", "worthlessness"], "intermediates": ["rejection_rule", "avoidance_rule"], "coping": ["withdrawal", "suppression"], "emotions": ["loneliness", "shame"], "behaviors": ["isolation", "social_avoidance"], "prompts": ["emotion_exploration", "belief_elicitation", "values_elicitation"]},
    "failure_identity_theme": {"beliefs": ["failure_identity", "worthlessness"], "intermediates": ["perfectionism_rule", "hopelessness_rule"], "coping": ["self_attack", "overplanning"], "emotions": ["shame", "hopelessness"], "behaviors": ["procrastination", "overworking"], "prompts": ["meaning_exploration", "cost_of_status_quo", "strengths_exceptions"]},
    "avoidance_pattern": {"beliefs": ["helplessness", "danger"], "intermediates": ["avoidance_rule"], "coping": ["avoidance"], "emotions": ["anxiety"], "behaviors": ["social_avoidance", "conflict_avoidance"], "prompts": ["coping_exploration", "discrepancy_elicitation", "action_readiness"]},
    "withdrawal_pattern": {"beliefs": ["unlovability", "helplessness"], "intermediates": ["emotional_inhibition_rule"], "coping": ["withdrawal"], "emotions": ["sadness", "loneliness"], "behaviors": ["isolation", "emotional_withdrawal"], "prompts": ["emotion_exploration", "coping_exploration", "future_projection"]},
    "perfectionism_pattern": {"beliefs": ["failure_identity", "worthlessness"], "intermediates": ["perfectionism_rule"], "coping": ["overplanning", "self_attack"], "emotions": ["anxiety", "shame"], "behaviors": ["overworking", "procrastination"], "prompts": ["thought_elicitation", "discrepancy_elicitation", "action_readiness"]},
    "rumination_pattern": {"beliefs": ["danger", "helplessness"], "intermediates": ["control_rule", "hopelessness_rule"], "coping": ["rumination"], "emotions": ["anxiety", "hopelessness"], "behaviors": ["checking_monitoring", "reassurance_loop"], "prompts": ["thought_elicitation", "coping_exploration", "strengths_exceptions"]},
    "self_attack_pattern": {"beliefs": ["worthlessness", "defectiveness"], "intermediates": ["perfectionism_rule", "self_silencing_rule"], "coping": ["self_attack"], "emotions": ["shame", "guilt"], "behaviors": ["procrastination", "emotional_withdrawal"], "prompts": ["belief_elicitation", "cost_of_status_quo", "values_elicitation"]},
    "overcontrol_pattern": {"beliefs": ["danger", "loss_of_control"], "intermediates": ["control_rule"], "coping": ["emotional_control", "overplanning"], "emotions": ["anxiety", "anger"], "behaviors": ["routine_control", "checking_monitoring"], "prompts": ["coping_exploration", "discrepancy_elicitation", "future_projection"]},
    "substance_use": {"beliefs": ["loss_of_control", "helplessness"], "intermediates": ["avoidance_rule", "hopelessness_rule"], "coping": ["substance_use_coping", "escape"], "emotions": ["shame", "numbness"], "behaviors": ["relapse_behavior", "emotional_withdrawal"], "prompts": ["cost_of_status_quo", "coping_exploration", "action_readiness"]},
    "relapse_risk": {"beliefs": ["loss_of_control", "failure_identity"], "intermediates": ["hopelessness_rule", "control_rule"], "coping": ["substance_use_coping", "rumination"], "emotions": ["anxiety", "shame"], "behaviors": ["checking_monitoring", "relapse_behavior"], "prompts": ["strengths_exceptions", "cost_of_status_quo", "action_readiness"]},
    "craving_escape": {"beliefs": ["helplessness", "loss_of_control"], "intermediates": ["avoidance_rule", "emotional_inhibition_rule"], "coping": ["escape", "substance_use_coping"], "emotions": ["anxiety", "numbness"], "behaviors": ["relapse_behavior", "emotional_withdrawal"], "prompts": ["coping_exploration", "strengths_exceptions", "action_readiness"]},
    "safety_hypervigilance": {"beliefs": ["danger", "mistrust"], "intermediates": ["control_rule", "mistrust_rule"], "coping": ["safety_behaviors", "emotional_control"], "emotions": ["anxiety"], "behaviors": ["checking_monitoring", "routine_control"], "prompts": ["thought_elicitation", "coping_exploration", "action_readiness"]},
    "control_loss_theme": {"beliefs": ["loss_of_control", "helplessness"], "intermediates": ["control_rule", "hopelessness_rule"], "coping": ["emotional_control", "overplanning"], "emotions": ["anxiety", "hopelessness"], "behaviors": ["routine_control", "checking_monitoring"], "prompts": ["belief_elicitation", "discrepancy_elicitation", "strengths_exceptions"]},
    "loneliness_exposure": {"beliefs": ["abandonment", "unlovability"], "intermediates": ["rejection_rule", "dependency_rule"], "coping": ["withdrawal", "reassurance_seeking"], "emotions": ["loneliness", "sadness"], "behaviors": ["isolation", "reassurance_loop"], "prompts": ["emotion_exploration", "values_elicitation", "future_projection"]},
}

SUBTOPIC_MAPS.update(
    {
        "depression_symptoms": {"beliefs": ["worthlessness", "helplessness"], "intermediates": ["hopelessness_rule", "emotional_inhibition_rule"], "coping": ["withdrawal", "suppression"], "emotions": ["sadness", "hopelessness"], "behaviors": ["isolation", "emotional_withdrawal"], "prompts": ["emotion_exploration", "cost_of_status_quo", "strengths_exceptions"]},
        "anxiety_symptoms": {"beliefs": ["danger", "helplessness"], "intermediates": ["avoidance_rule", "control_rule"], "coping": ["avoidance", "safety_behaviors"], "emotions": ["anxiety", "fear"], "behaviors": ["checking_monitoring", "social_avoidance"], "prompts": ["thought_elicitation", "coping_exploration", "action_readiness"]},
        "stress_overload": {"beliefs": ["helplessness", "loss_of_control"], "intermediates": ["control_rule", "hopelessness_rule"], "coping": ["overplanning", "suppression"], "emotions": ["anxiety", "irritability"], "behaviors": ["procrastination", "routine_control"], "prompts": ["emotion_exploration", "coping_exploration", "discrepancy_elicitation"]},
        "trauma_response": {"beliefs": ["danger", "mistrust"], "intermediates": ["mistrust_rule", "avoidance_rule"], "coping": ["safety_behaviors", "suppression"], "emotions": ["fear", "numbness"], "behaviors": ["checking_monitoring", "emotional_withdrawal"], "prompts": ["emotion_exploration", "coping_exploration", "history_linking"]},
        "emotion_dysregulation": {"beliefs": ["loss_of_control", "helplessness"], "intermediates": ["control_rule", "hopelessness_rule"], "coping": ["emotional_control", "escape"], "emotions": ["anxiety", "anger"], "behaviors": ["relapse_behavior", "emotional_withdrawal"], "prompts": ["thought_elicitation", "coping_exploration", "action_readiness"]},
        "therapy_history": {"beliefs": ["helplessness"], "intermediates": ["dependency_rule"], "coping": ["help_seeking"], "emotions": ["anxiety"], "behaviors": ["reassurance_loop"], "prompts": ["strengths_exceptions", "values_elicitation", "action_readiness"]},
        "psychiatric_medication": {"beliefs": ["helplessness", "loss_of_control"], "intermediates": ["dependency_rule", "control_rule"], "coping": ["help_seeking", "overplanning"], "emotions": ["anxiety"], "behaviors": ["routine_control", "checking_monitoring"], "prompts": ["coping_exploration", "strengths_exceptions", "action_readiness"]},
        "insurance_costs": {"beliefs": ["danger", "helplessness"], "intermediates": ["hopelessness_rule", "overresponsibility_rule"], "coping": ["avoidance", "overplanning"], "emotions": ["anxiety", "hopelessness"], "behaviors": ["checking_monitoring", "procrastination"], "prompts": ["cost_of_status_quo", "coping_exploration", "action_readiness"]},
        "compensation_support": {"beliefs": ["worthlessness", "danger"], "intermediates": ["mistrust_rule", "overresponsibility_rule"], "coping": ["avoidance", "people_pleasing"], "emotions": ["anxiety", "anger"], "behaviors": ["appeasement", "checking_monitoring"], "prompts": ["meaning_exploration", "values_elicitation", "action_readiness"]},
        "sexuality_identity": {"beliefs": ["shame_identity", "unlovability"], "intermediates": ["self_silencing_rule", "rejection_rule"], "coping": ["suppression", "withdrawal"], "emotions": ["shame", "anxiety"], "behaviors": ["emotional_withdrawal", "social_avoidance"], "prompts": ["belief_elicitation", "emotion_exploration", "values_elicitation"]},
        "independence_theme": {"beliefs": ["helplessness", "worthlessness"], "intermediates": ["dependency_rule", "control_rule"], "coping": ["people_pleasing", "overplanning"], "emotions": ["anxiety", "shame"], "behaviors": ["appeasement", "routine_control"], "prompts": ["meaning_exploration", "discrepancy_elicitation", "future_projection"]},
        "competence_theme": {"beliefs": ["failure_identity", "helplessness"], "intermediates": ["perfectionism_rule", "hopelessness_rule"], "coping": ["self_attack", "overplanning"], "emotions": ["shame", "anxiety"], "behaviors": ["procrastination", "overworking"], "prompts": ["belief_elicitation", "strengths_exceptions", "action_readiness"]},
        "personal_growth_theme": {"beliefs": ["worthlessness"], "intermediates": ["hopelessness_rule"], "coping": ["activity_scheduling", "help_seeking"], "emotions": ["anxiety"], "behaviors": ["routine_control"], "prompts": ["values_elicitation", "future_projection", "action_readiness"]},
        "friendship_strain": {"beliefs": ["unlovability", "mistrust"], "intermediates": ["rejection_rule", "mistrust_rule"], "coping": ["withdrawal", "people_pleasing"], "emotions": ["shame", "loneliness"], "behaviors": ["isolation", "appeasement"], "prompts": ["emotion_exploration", "meaning_exploration", "future_projection"]},
        "peer_rejection": {"beliefs": ["unlovability", "worthlessness"], "intermediates": ["rejection_rule", "avoidance_rule"], "coping": ["withdrawal", "self_attack"], "emotions": ["shame", "loneliness"], "behaviors": ["social_avoidance", "isolation"], "prompts": ["belief_elicitation", "history_linking", "future_projection"]},
        "community_disconnection": {"beliefs": ["unlovability", "worthlessness"], "intermediates": ["rejection_rule", "avoidance_rule"], "coping": ["withdrawal", "suppression"], "emotions": ["loneliness", "sadness"], "behaviors": ["isolation", "social_avoidance"], "prompts": ["emotion_exploration", "values_elicitation", "future_projection"]},
        "roommate_stress": {"beliefs": ["danger", "mistrust"], "intermediates": ["avoidance_rule", "mistrust_rule"], "coping": ["avoidance", "people_pleasing"], "emotions": ["anxiety", "anger"], "behaviors": ["conflict_avoidance", "appeasement"], "prompts": ["coping_exploration", "meaning_exploration", "action_readiness"]},
        "social_isolation": {"beliefs": ["unlovability", "worthlessness"], "intermediates": ["rejection_rule", "hopelessness_rule"], "coping": ["withdrawal", "suppression"], "emotions": ["loneliness", "hopelessness"], "behaviors": ["isolation", "emotional_withdrawal"], "prompts": ["emotion_exploration", "cost_of_status_quo", "future_projection"]},
        "caretaking_roles": {"beliefs": ["worthlessness", "helplessness"], "intermediates": ["overresponsibility_rule", "self_silencing_rule"], "coping": ["people_pleasing", "suppression"], "emotions": ["guilt", "anxiety"], "behaviors": ["appeasement", "overworking"], "prompts": ["meaning_exploration", "discrepancy_elicitation", "values_elicitation"]},
        "moving_stress": {"beliefs": ["helplessness", "abandonment"], "intermediates": ["avoidance_rule", "control_rule"], "coping": ["overplanning", "suppression"], "emotions": ["anxiety", "sadness"], "behaviors": ["routine_control", "emotional_withdrawal"], "prompts": ["emotion_exploration", "history_linking", "future_projection"]},
        "household_duties": {"beliefs": ["worthlessness", "helplessness"], "intermediates": ["overresponsibility_rule", "perfectionism_rule"], "coping": ["overplanning", "self_attack"], "emotions": ["anxiety", "irritability"], "behaviors": ["overworking", "procrastination"], "prompts": ["coping_exploration", "discrepancy_elicitation", "action_readiness"]},
        "domestic_stress": {"beliefs": ["danger", "helplessness"], "intermediates": ["avoidance_rule", "control_rule"], "coping": ["suppression", "people_pleasing"], "emotions": ["anxiety", "anger"], "behaviors": ["conflict_avoidance", "emotional_withdrawal"], "prompts": ["emotion_exploration", "coping_exploration", "values_elicitation"]},
        "living_arrangement_tension": {"beliefs": ["unlovability", "mistrust"], "intermediates": ["avoidance_rule", "mistrust_rule"], "coping": ["withdrawal", "people_pleasing"], "emotions": ["anxiety", "shame"], "behaviors": ["social_avoidance", "appeasement"], "prompts": ["meaning_exploration", "coping_exploration", "future_projection"]},
        "home_instability": {"beliefs": ["danger", "abandonment"], "intermediates": ["control_rule", "dependency_rule"], "coping": ["overplanning", "help_seeking"], "emotions": ["fear", "hopelessness"], "behaviors": ["checking_monitoring", "routine_control"], "prompts": ["emotion_exploration", "cost_of_status_quo", "action_readiness"]},
        "missed_goodbye": {"beliefs": ["worthlessness", "abandonment"], "intermediates": ["hopelessness_rule", "emotional_inhibition_rule"], "coping": ["rumination", "withdrawal"], "emotions": ["guilt", "sadness"], "behaviors": ["isolation", "emotional_withdrawal"], "prompts": ["emotion_exploration", "history_linking", "future_projection"]},
        "role_status_loss": {"beliefs": ["failure_identity", "worthlessness"], "intermediates": ["hopelessness_rule", "perfectionism_rule"], "coping": ["withdrawal", "self_attack"], "emotions": ["shame", "sadness"], "behaviors": ["isolation", "procrastination"], "prompts": ["meaning_exploration", "values_elicitation", "future_projection"]},
    }
)


BELIEF_TO_INTERMEDIATE = {
    "helplessness": ["avoidance_rule", "hopelessness_rule", "dependency_rule"],
    "unlovability": ["rejection_rule", "self_silencing_rule"],
    "worthlessness": ["hopelessness_rule", "self_silencing_rule"],
    "defectiveness": ["rejection_rule", "emotional_inhibition_rule"],
    "danger": ["avoidance_rule", "control_rule", "mistrust_rule"],
    "mistrust": ["mistrust_rule", "control_rule"],
    "abandonment": ["dependency_rule", "rejection_rule"],
    "failure_identity": ["perfectionism_rule", "hopelessness_rule"],
    "shame_identity": ["self_silencing_rule", "rejection_rule"],
    "loss_of_control": ["control_rule", "hopelessness_rule"],
}


INTERMEDIATE_TO_COPING = {
    "rejection_rule": ["withdrawal", "people_pleasing", "suppression"],
    "avoidance_rule": ["avoidance", "escape", "safety_behaviors"],
    "perfectionism_rule": ["overplanning", "self_attack"],
    "control_rule": ["emotional_control", "overplanning", "safety_behaviors"],
    "self_silencing_rule": ["suppression", "people_pleasing"],
    "mistrust_rule": ["withdrawal", "safety_behaviors"],
    "hopelessness_rule": ["withdrawal", "escape", "rumination"],
    "dependency_rule": ["reassurance_seeking", "people_pleasing", "help_seeking"],
    "overresponsibility_rule": ["overplanning", "people_pleasing"],
    "emotional_inhibition_rule": ["suppression", "emotional_control"],
}


PROMPT_LINKS = {
    "helplessness": ["belief_elicitation", "cost_of_status_quo", "strengths_exceptions"],
    "unlovability": ["belief_elicitation", "emotion_exploration", "future_projection"],
    "worthlessness": ["belief_elicitation", "values_elicitation", "strengths_exceptions"],
    "defectiveness": ["meaning_exploration", "belief_elicitation", "future_projection"],
    "danger": ["thought_elicitation", "coping_exploration", "action_readiness"],
    "mistrust": ["meaning_exploration", "history_linking", "values_elicitation"],
    "abandonment": ["emotion_exploration", "belief_elicitation", "future_projection"],
    "failure_identity": ["meaning_exploration", "discrepancy_elicitation", "strengths_exceptions"],
    "shame_identity": ["emotion_exploration", "belief_elicitation", "values_elicitation"],
    "loss_of_control": ["belief_elicitation", "coping_exploration", "action_readiness"],
    "avoidance": ["coping_exploration", "discrepancy_elicitation", "action_readiness"],
    "withdrawal": ["emotion_exploration", "coping_exploration", "future_projection"],
    "escape": ["coping_exploration", "cost_of_status_quo", "strengths_exceptions"],
    "suppression": ["emotion_exploration", "meaning_exploration", "values_elicitation"],
    "people_pleasing": ["meaning_exploration", "discrepancy_elicitation", "values_elicitation"],
    "reassurance_seeking": ["thought_elicitation", "coping_exploration", "strengths_exceptions"],
    "overplanning": ["coping_exploration", "discrepancy_elicitation", "action_readiness"],
    "self_attack": ["belief_elicitation", "cost_of_status_quo", "strengths_exceptions"],
    "substance_use_coping": ["cost_of_status_quo", "coping_exploration", "action_readiness"],
    "emotional_control": ["coping_exploration", "discrepancy_elicitation", "future_projection"],
    "safety_behaviors": ["thought_elicitation", "coping_exploration", "action_readiness"],
    "rumination": ["thought_elicitation", "coping_exploration", "strengths_exceptions"],
    "help_seeking": ["strengths_exceptions", "values_elicitation", "action_readiness"],
    "activity_scheduling": ["strengths_exceptions", "future_projection", "action_readiness"],
    "anxiety": ["emotion_exploration", "thought_elicitation", "coping_exploration"],
    "shame": ["emotion_exploration", "belief_elicitation", "meaning_exploration"],
    "sadness": ["emotion_exploration", "history_linking", "future_projection"],
    "anger": ["emotion_exploration", "meaning_exploration", "values_elicitation"],
    "guilt": ["emotion_exploration", "belief_elicitation", "values_elicitation"],
    "loneliness": ["emotion_exploration", "history_linking", "future_projection"],
    "hopelessness": ["cost_of_status_quo", "strengths_exceptions", "action_readiness"],
    "numbness": ["emotion_exploration", "coping_exploration", "future_projection"],
}


def slug(label: str) -> str:
    return label.lower().replace(" / ", "_").replace(" ", "_").replace("-", "_")


def add_node(nodes: dict[str, dict[str, Any]], node_id: str, **attrs: Any) -> None:
    if node_id in nodes:
        return
    nodes[node_id] = {"id": node_id, **attrs}


def add_edge(edges: set[tuple[str, str, str, float]], source: str, target: str, relation: str, weight: float = 1.0) -> None:
    edges.add((source, target, relation, weight))


def build_graph_spec() -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: set[tuple[str, str, str, float]] = set()

    for topic_id, topic in SUPER_TOPICS.items():
        add_node(
            nodes,
            topic_id,
            type="super_topic",
            label=topic["label"],
            description=topic["description"],
            aliases=topic["aliases"],
        )

    for subtopic_id, subtopic in SUBTOPICS.items():
        add_node(
            nodes,
            subtopic_id,
            type="sub_topic",
            label=subtopic["label"],
            aliases=subtopic["aliases"],
        )

    for belief_id, description in BELIEF_CLUSTERS.items():
        add_node(
            nodes,
            belief_id,
            type="core_belief_cluster",
            label=belief_id.replace("_", " ").title(),
            description=description,
        )

    for intermediate_id, description in INTERMEDIATE_TYPES.items():
        add_node(
            nodes,
            intermediate_id,
            type="intermediate_belief_type",
            label=intermediate_id.replace("_", " ").title(),
            description=description,
        )

    for coping_id, description in COPING_TYPES.items():
        add_node(
            nodes,
            coping_id,
            type="coping_type",
            label=coping_id.replace("_", " ").title(),
            description=description,
        )

    for emotion_id, description in EMOTION_CLUSTERS.items():
        add_node(
            nodes,
            emotion_id,
            type="emotion_cluster",
            label=emotion_id.title(),
            description=description,
        )

    for behavior_id, description in BEHAVIOR_PATTERNS.items():
        add_node(
            nodes,
            behavior_id,
            type="behavior_pattern",
            label=behavior_id.replace("_", " ").title(),
            description=description,
        )

    for prompt_id, templates in PROMPT_TEMPLATES.items():
        add_node(
            nodes,
            prompt_id,
            type="prompt_type",
            label=prompt_id.replace("_", " ").title(),
            prompt_templates=templates,
        )

    for topic_id, topic in SUPER_TOPICS.items():
        for subtopic_id in topic["subtopics"]:
            add_edge(edges, topic_id, subtopic_id, "contains")

    for subtopic_id, mapping in SUBTOPIC_MAPS.items():
        for belief_id in mapping["beliefs"]:
            add_edge(edges, subtopic_id, belief_id, "associated_with", 0.9)
        for intermediate_id in mapping["intermediates"]:
            add_edge(edges, subtopic_id, intermediate_id, "associated_with", 0.8)
        for coping_id in mapping["coping"]:
            add_edge(edges, subtopic_id, coping_id, "associated_with", 0.8)
        for emotion_id in mapping["emotions"]:
            add_edge(edges, subtopic_id, emotion_id, "evokes", 0.75)
        for behavior_id in mapping["behaviors"]:
            add_edge(edges, subtopic_id, behavior_id, "manifests_as", 0.75)
        for prompt_id in mapping["prompts"]:
            add_edge(edges, subtopic_id, prompt_id, "suggests_prompt_type", 0.7)

    for belief_id, intermediate_ids in BELIEF_TO_INTERMEDIATE.items():
        for intermediate_id in intermediate_ids:
            add_edge(edges, belief_id, intermediate_id, "leads_to", 0.7)

    for intermediate_id, coping_ids in INTERMEDIATE_TO_COPING.items():
        for coping_id in coping_ids:
            add_edge(edges, intermediate_id, coping_id, "maintained_by", 0.7)

    for source_id, prompt_ids in PROMPT_LINKS.items():
        for prompt_id in prompt_ids:
            add_edge(edges, source_id, prompt_id, "suggests_prompt_type", 0.65)

    spec_edges = [
        {"source": source, "target": target, "relation": relation, "weight": weight}
        for source, target, relation, weight in sorted(edges)
    ]

    return {
        "metadata": {
            "name": "mind_voyager_generalized_topic_graph",
            "description": (
                "Generalized CBT-oriented topic ontology for therapy dialogue retrieval, "
                "covering shared topic domains, belief clusters, intermediate beliefs, "
                "coping styles, emotional themes, behavior patterns, and prompt types."
            ),
            "node_count": len(nodes),
            "edge_count": len(spec_edges),
        },
        "nodes": [nodes[node_id] for node_id in sorted(nodes)],
        "edges": spec_edges,
    }


def export_graph(spec: dict[str, Any], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    spec_path = output_dir / "topic_graph.json"
    spec_path.write_text(json.dumps(spec, indent=2))
    written.append(spec_path)

    summary = {
        "metadata": spec["metadata"],
        "node_types": {},
        "edge_relations": {},
    }

    for node in spec["nodes"]:
        node_type = node["type"]
        summary["node_types"][node_type] = summary["node_types"].get(node_type, 0) + 1
    for edge in spec["edges"]:
        relation = edge["relation"]
        summary["edge_relations"][relation] = summary["edge_relations"].get(relation, 0) + 1

    summary_path = output_dir / "topic_graph_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    written.append(summary_path)

    try:
        import networkx as nx
    except ImportError:
        return written

    def graphml_safe_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
        safe: dict[str, Any] = {}
        for key, value in attrs.items():
            if isinstance(value, (list, dict)):
                safe[key] = json.dumps(value, ensure_ascii=False)
            else:
                safe[key] = value
        return safe

    graph = nx.DiGraph()
    for node in spec["nodes"]:
        graph.add_node(
            node["id"],
            **graphml_safe_attrs({k: v for k, v in node.items() if k != "id"}),
        )
    for edge in spec["edges"]:
        graph.add_edge(
            edge["source"],
            edge["target"],
            relation=edge["relation"],
            weight=edge["weight"],
        )

    graphml_path = output_dir / "topic_graph.graphml"
    nx.write_graphml(graph, graphml_path)
    written.append(graphml_path)
    return written


def neo4j_labels(node_type: str) -> list[str]:
    parts = node_type.split("_")
    type_label = "".join(part.capitalize() for part in parts)
    return ["TopicGraphNode", type_label]


def export_to_neo4j(
    spec: dict[str, Any],
    uri: str,
    user: str,
    password: str,
    database: str,
    clear_existing: bool = False,
) -> dict[str, Any]:
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise RuntimeError(
            "Neo4j export requires the 'neo4j' package. Install it with 'pip install neo4j'."
        ) from exc

    driver = GraphDatabase.driver(uri, auth=(user, password))

    def create_constraint(tx: Any) -> None:
        tx.run(
            """
            CREATE CONSTRAINT topic_graph_node_id IF NOT EXISTS
            FOR (n:TopicGraphNode)
            REQUIRE n.id IS UNIQUE
            """
        )

    def clear_graph(tx: Any) -> None:
        tx.run("MATCH (n:TopicGraphNode) DETACH DELETE n")

    def upsert_nodes(tx: Any) -> None:
        for node in spec["nodes"]:
            labels = ":".join(neo4j_labels(node["type"]))
            properties = {k: v for k, v in node.items() if k != "id"}
            tx.run(
                f"""
                MERGE (n:{labels} {{id: $id}})
                SET n += $properties
                """,
                id=node["id"],
                properties=properties,
            )

    def upsert_edges(tx: Any) -> None:
        for edge in spec["edges"]:
            relation_type = edge["relation"].upper()
            tx.run(
                f"""
                MATCH (source:TopicGraphNode {{id: $source_id}})
                MATCH (target:TopicGraphNode {{id: $target_id}})
                MERGE (source)-[rel:{relation_type}]->(target)
                SET rel.relation = $relation,
                    rel.weight = $weight
                """,
                source_id=edge["source"],
                target_id=edge["target"],
                relation=edge["relation"],
                weight=edge["weight"],
            )

    try:
        with driver.session(database=database) as session:
            session.execute_write(create_constraint)
            if clear_existing:
                session.execute_write(clear_graph)
            session.execute_write(upsert_nodes)
            session.execute_write(upsert_edges)
    finally:
        driver.close()

    return {
        "uri": uri,
        "database": database,
        "node_count": len(spec["nodes"]),
        "edge_count": len(spec["edges"]),
        "cleared_existing": clear_existing,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the generalized therapy topic graph ontology")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT),
        help="Directory where the topic graph files will be written",
    )
    parser.add_argument(
        "--neo4j-uri",
        help="Optional Neo4j URI to also upload the graph, for example bolt://localhost:7687",
    )
    parser.add_argument(
        "--neo4j-user",
        default="neo4j",
        help="Neo4j username used with --neo4j-uri",
    )
    parser.add_argument(
        "--neo4j-password",
        help="Neo4j password used with --neo4j-uri",
    )
    parser.add_argument(
        "--neo4j-database",
        default="neo4j",
        help="Neo4j database name used with --neo4j-uri",
    )
    parser.add_argument(
        "--neo4j-clear",
        action="store_true",
        help="Delete existing TopicGraphNode nodes before uploading the new graph",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    spec = build_graph_spec()
    written = export_graph(spec, Path(args.output_dir))
    result: dict[str, Any] = {
        "written_files": [str(path) for path in written],
        "metadata": spec["metadata"],
    }

    if args.neo4j_uri:
        if not args.neo4j_password:
            raise ValueError("--neo4j-password is required when --neo4j-uri is provided")
        result["neo4j"] = export_to_neo4j(
            spec=spec,
            uri=args.neo4j_uri,
            user=args.neo4j_user,
            password=args.neo4j_password,
            database=args.neo4j_database,
            clear_existing=args.neo4j_clear,
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
