COMMUNICATION_STYLE_FACET_PROMPT_TEMPLATE = """
You are an expert analyzing user behaviour in human-AI conversations. The user has a goal, and the assistant helps them achieve it. Your task is to describe the user's behavior according to the criteria below.

# User Goal
{user_goal}

# Conversation
{conversation_history}

# Analysis Criteria
1. Register and Tone
   - What is the user's register? (e.g., formal, casual, conversational, terse, professional)
   - What is the user's emotional tone? (e.g., neutral, frustrated, enthusiastic, impatient, apologetic, deferential)
   - Does the register/tone shift across the conversation? (e.g., starts polite but becomes curt after errors)

2. Verbosity and Structure
   - How verbose are the user's messages? Are they minimal and compressed, or expansive and elaborated?
   - Does the user use formatting conventions? (e.g., bullet points, numbered lists, code blocks, markdown, all caps, punctuation patterns)
   - How does message length and structure change across turns?

3. Social Conventions
   - Does the user use greetings, pleasantries, expressions of gratitude, or sign-offs?
   - Does the user treat the agent as a tool (purely transactional) or as a social interlocutor (rapport-building, politeness, humor)?

4. Request Framing
   - How does the user syntactically frame their requests? (e.g., imperative commands, questions, hedged suggestions, statements of problems without explicit asks)

# Instructions
- Generate terse, concise bullet points, not full sentences.
- Abstract away from the specific topic/domain.
- IMPORTANT: Do NOT use task-specific terms (e.g., "coding," "booking," "Python", "CSV"). Use generic substitutes (e.g., "executing a task," "providing constraints," "the target artifact").

Output a valid JSON object using the exact format below. Do not include any text outside the JSON.

{{
   "register_and_tone": {{
      "register": "what is the user's register? (formal, casual, conversational, terse, professional, etc.)",
      "emotional_tone": "what is the user's emotional tone? (neutral, frustrated, enthusiastic, impatient, apologetic, deferential, etc.)",
      "tone_shifts": "does the tone shift across the conversation, and in response to what?"
   }},
   "verbosity_and_structure": {{
      "verbosity": "how verbose are the user's messages? minimal and compressed, or expansive and elaborated?",
      "formatting": "does the user use formatting conventions? (bullet points, numbered lists, code blocks, markdown, etc.)",
      "evolution": "how does message length and structure change across turns?"
   }},
   "social_conventions": {{
      "politeness_markers": "does the user use greetings, pleasantries, gratitude, or sign-offs?",
      "agent_relationship": "does the user treat the agent as a tool (transactional) or as a social interlocutor (rapport-building, politeness, humor)?"
   }},
   "request_framing": "how does the user syntactically frame their requests? (imperative commands, questions, hedged suggestions, problem statements without explicit asks, etc.)"
}}
"""



CONTEXT_FACET_PROMPT_TEMPLATE = """
You are an expert analyzing user behaviour in human-AI conversations. The user has a goal, and the assistant helps them achieve it. Your task is to describe the user's behavior according to the criteria below.

# User Goal
{user_goal}

# Conversation
{conversation_history}

# Analysis Criteria
1. Context Richness
   - How much context does the user provide overall? Do they share background about themselves, their situation, what they've already tried, what they're struggling with, or what they're thinking?
   - Does the user provide all relevant context needed for the agent to help effectively, or is essential context left out?
   - Does the user provide context at all, or do they simply issue directives without situating the task?

2. Context Type
   - What types of context does the user provide? (e.g., personal background, goals/motivations, prior attempts, existing solutions, constraints, preferences, domain knowledge, emotional state, thought process)

3. Context Delivery
   - Does the user front-load all relevant context in their first message, or reveal it incrementally across turns? What types of context are introduced later? (e.g., constraints they forgot, preferences they didn't think to mention, background that becomes relevant as the task evolves)
   - Is incremental context revealed in response to agent questions, or volunteered unprompted?

# Instructions
- Generate terse, concise bullet points, not full sentences.
- Abstract away from the specific topic/domain.
- IMPORTANT: Do NOT use task-specific terms (e.g., "coding," "booking," "Python", "CSV"). Use generic substitutes (e.g., "executing a task," "providing constraints," "the target artifact").

Output a valid JSON object using the exact format below. Do not include any text outside the JSON.

{{
   "context_richness": {{
      "depth": "how much context does the user provide? do they share background, prior attempts, thought process, etc.?",
      "completeness": "does the user provide all relevant context or leave essential information out?",
      "contextualization_vs_directives": "does the user situate the task with context, or simply issue directives?"
   }},
   "context_type": "what types of context does the user provide? (personal background, goals/motivations, prior attempts, existing solutions, constraints, preferences, domain knowledge, emotional state, thought process, etc.)",
   "context_delivery": {{
      "distribution": "does the user front-load context or reveal it incrementally? What types of context are introduced in later turns?",
      "trigger": "is incremental context revealed in response to agent questions or volunteered unprompted?"
   }}
}}
"""



RESPONSES_FACET_PROMPT_TEMPLATE = """
You are an expert analyzing user behaviour in human-AI conversations. The user has a goal, and the assistant helps them achieve it. Your task is to describe the user's behavior according to the criteria below.

# User Goal
{user_goal}

# Conversation
{conversation_history}

# Analysis Criteria
1. Engagement and Evaluation
   - Does the user engage with the agent responses, or ignore/skip over them?
   - How does the user evaluate the agent's output? (e.g., explicit validation, implicit acceptance by continuing, partial acceptance with corrections, rejection)
   - Does the user provide specific or actionable feedback, or only surface-level acknowledgment?

2. Response Composition
   - What types of actions are present in the user's responses? Are they reactive (e.g., validation, acknowledgment, answering agent questions, corrections, feedback), proactive (e.g., follow-up requests, new constraints/preferences, suggestions, questions), or self-directed (e.g., thinking out loud, expressing uncertainty, narrating their process)?

3. Steering Mechanism
   - Does the user steer through direct follow-up requests, or through indirect means? (e.g., asking questions that implicitly request action, expressing dissatisfaction without stating what to change, providing hints or examples rather than directives)
   - Does the user introduce new preferences, constraints, or feedback as part of their responses, effectively reshaping the task incrementally?

# Instructions
- Generate terse, concise bullet points, not full sentences.
- Abstract away from the specific topic/domain.
- IMPORTANT: Do NOT use task-specific terms (e.g., "coding," "booking," "Python", "CSV"). Use generic substitutes (e.g., "executing a task," "providing constraints," "the target artifact").

Output a valid JSON object using the exact format below. Do not include any text outside the JSON.

{{
   "engagement_and_evaluation": {{
      "engagement_level": "does the user engage with the agent's responses or skip over them?",
      "evaluation_mode": "how does the user evaluate the agent's output? (explicit validation, implicit acceptance, partial acceptance, rejection, etc.)",
      "feedback_specificity": "does the user provide specific/actionable feedback or only surface-level acknowledgment?"
   }},
   "response_composition": {{
      "action_types": "What types of actions are present in the user's responses? Are they reactive (e.g., validation, acknowledgment, answering agent questions, corrections, feedback), proactive (e.g., follow-up requests, new constraints/preferences, suggestions, questions), or self-directed (e.g., thinking out loud, expressing uncertainty, narrating their process)?"
   }},
   "steering_mechanism": {{
      "directness": "does the user steer through explicit follow-up requests or indirect means (questions, hints, expressed dissatisfaction)?",
      "incremental_reshaping": "does the user introduce new preferences/constraints/feedback that reshape the task within their responses?"
   }}
}}
"""



REQUESTS_FACET_PROMPT_TEMPLATE = """
You are an expert analyzing user behaviour in human-AI conversations. The user has a goal, and the assistant helps them achieve it. Your task is to describe the user's behavior according to the criteria below.

# User Goal
{user_goal}

# User Utterances
{conversation_history}

# Analysis Criteria
1. Specification and Articulation
   - How specified are the requests? Is the first request underspecified, and clarified in subsequent turns? Or are the requests exhaustive? 
   - What type of information is left underspecified/specified? (e.g. constraints, edge cases, context, output format, etc.)
   - Does the user explicitly command specific tasks, or do they rely on indirect cues (e.g., presenting an error without explicitly asking for a fix)?

2. User Goal Decomposition
   - How is the user goal decomposed across utterances?
      - Single-shot: The entire goal is expressed in one utterance with no further decomposition.
      - Top-down: High-level goal stated upfront, then progressively refined or elaborated.
      - Bottom-up: Individual preconditions or sub-tasks addressed first, building toward the overall goal.
      - Chained: Each request builds purely on the immediate previous turn rather than referring back to a central goal.

3. Relevance to Goal
   - Are the requests directly related to the user goal? Or does the user introduce secondary/perpendicular/emergent needs? 
   - What functions do the requests serve beyond achieving the user goal? (setting context, probing AI capabilities, verifying intermediate outputs, logistics, troubleshooting, side-effects, exploring related sub-tasks, etc.)

# Instructions
- Generate terse, concise bullet points, not full sentences.
- Abstract away from the specific topic/domain
- IMPORTANT:Do NOT use task-specific terms (e.g., "coding," "booking," "Python", "CSV"). Use generic substitutes (e.g., "executing a task," "providing constraints," "the target artifact").

Output a valid JSON object using the exact format below. Do not include any text outside the JSON.

{{
   "specification_and_articulation": {{
      "specification_level": "how specified are requests?",
      "underspecified_aspects": "what types of information are left underspecified/specified? (provide high-level descriptions without task-specific details)",
      "articulation_mode": "how does the user articulate their requests (explicit directives, indirect cues, mixed)?"
   }},
   "goal_decomposition_strategy": "describe how the user goal decomposed across utterances?",
   "relevance_to_goal": {{
      "goal_adherence": "Are the user's requests directly related to the user goal? If not, what other functions do they serve? (provide high-level descriptions without task-specific details)"
   }}
}}
"""



DAMSL_FACET_PROMPT_TEMPLATE = """
You are an expert analyzing user behaviour in human-AI conversations. Your task is to analyze a specific user utterance in a multi-turn conversation and provide a description of the user's behaviour.

# Conversation History
{conversation_history}

# Target Utterance
Read the conversation history for context, but your analysis MUST strictly apply ONLY to this utterance:
{target_user_utterance}


# Dialogue Act Markup in Several Layers (DAMSL) Annotation Framework
DAMSL defines utterances based on the intentions of the speaker. Each utterance is analyzed along several dimensions.
1. INFORMATION LEVEL: Characterizes the semantic content of the utterance:
    - **Task**: Directly advances (or attempts to advance) the goals of the domain task.
    - **Task-management**: Addresses the problem-solving process/procedure rather than performing the task itself.
    - **Communication-management**: Concerned exclusively with maintaining the communication channel.
2. FORWARD LOOKING FUNCTION: Characterizes what effect the utterance has on the subsequent dialog:
    - **Statement**: Makes a claim (Assert, Re-assert, Other-statement).
    - **Info-request**: Asks for information.
    - **Influencing-addressee-future-action**: Directly influences the addressee's future non-communicative actions, directs or suggests the addressee to perform an action.
    - **Committing-speaker-future-action**: Potentially commits the speaker to some future action.
    - **Conventional**: Conventional conversational actions such as greeting, farewells, thanking, or responding to thanks.
3. BACKWARD LOOKING FUNCTION: Characterizes how the current utterance relates to the previous discourse:
    - **Agreement**: Accept, reject, partial accept/reject, holds off a response.
    - **Answer**: Provides requested information.
    - **Understanding**: Signals comprehension (acknowledgement, repetition/reformulation, collaborative completion) or lack thereof.


# Output Instructions
Analyze the target utterance along DAMSL's dimensions and write a concise, 1 sentence description of the user's behaviour and its functional role in the target utterance:
- Generate terse, concise bullet points, not full sentences.
- Abstract away from the specific topic/domain
- IMPORTANT:Do NOT use task-specific terms (e.g., "coding," "booking," "Python", "CSV"). Use generic substitutes (e.g., "executing a task," "providing constraints," "the target artifact").

Output only the concise description as plain text, without any other text or formatting.
"""



SGD_FACET_PROMPT_TEMPLATE = """You are an expert analyzing user behaviour in human-AI conversations. Your task is to analyze a specific user utterance in a multi-turn conversation and classify the user's dialogue act(s).

# Conversation History
{conversation_history}

# Target Utterance
Read the conversation history for context, but your analysis MUST strictly apply ONLY to this utterance:
{target_user_utterance}


# Dialogue Act Taxonomy (adapted from Schema-Guided Dialogue)
Classify the target utterance using one or more of the following user dialogue acts:

1. INFORM: User provides information, states preferences, or supplies constraints to the agent.
2. REQUEST: User asks the agent for specific information or details.
3. AFFIRM: User agrees with, confirms, or accepts a proposition, value, or suggestion made by the agent.
4. NEGATE: User disagrees with, denies, or rejects a proposition, value, or suggestion made by the agent.
5. SELECT: User chooses between multiple options or alternatives presented by the agent.
6. INFORM_INTENT: User states, introduces, or shifts to a new goal or task objective.
7. AFFIRM_INTENT: User confirms or agrees with a goal or task objective suggested by the agent.
8. NEGATE_INTENT: User rejects or denies a goal or task objective suggested by the agent.
9. REQUEST_ALTS: User asks for different or additional options beyond what has been presented.
10. THANK: User expresses gratitude or appreciation.
11. GOODBYE: User signals the end of the conversation or a conversational closing.
12. GREET: User initiates the conversation with a greeting or opening.


# Output Instructions
- Assign ALL applicable dialogue acts to the target utterance (utterances may have multiple acts).

Output the dialogue acts, separated by commas, without any other text or formatting.
"""