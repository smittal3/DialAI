INSERT INTO conversations_history (start_time, end_time, duration_seconds, conversation_history) 
VALUES (
    '2025-02-04 13:25:34.567890',
    '2025-02-04 13:27:15.123456',
    101,
    '[
        {
            "role": "assistant",
            "message": "Hello, I''m calling from FundANeed Foundation. Am I speaking with Mr. Reddy?"
        },
        {
            "role": "user",
            "message": "Yes, but I was about to call your office. None of us have received the funds."
        },
        {
            "role": "assistant",
            "message": "Could you elaborate? Who do you mean by ''us''?"
        },
        {
            "role": "user",
            "message": "Everyone who applied through the SMS camp at Krishna Colony last month. At least 15 families I know. We all got approval messages but no money. The school registration deadline is tomorrow."
        },
        {
            "role": "assistant",
            "message": "I see. Did everyone receive the approval confirmation by SMS?"
        },
        {
            "role": "user",
            "message": "Yes, we all got messages saying ''Your application is approved'' but when we click the link to check status, it shows error. The local coordinator said system is updating but it''s been 3 weeks."
        },
        {
            "role": "assistant",
            "message": "Did anyone from your group receive the funds?"
        },
        {
            "role": "user",
            "message": "No one. We''re all middle-class parents, we don''t have extra savings for fees. Some children already missed first week of classes."
        }
    ]'
); 