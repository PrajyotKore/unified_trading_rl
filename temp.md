flowchart LR
    %% Entry Channels
    subgraph Channels
        A1[Phone / PSTN]
        A2[Web & Mobile Apps]
        A3[SMS / WhatsApp]
        A4[Provider & Member Portals]
    end

    %% Contact Center / Entry Platform
    subgraph Contact_Center_Platform[Contact Center & Channel Platform]
        B1[Smart IVR\n(ASR + NLU + Auth)]
        B2[Omnichannel Router]
    end

    %% Experience & AI Orchestration
    subgraph Experience_AI[Experience & AI Layer]
        C0[Conversation Orchestrator\n(Session + Business Rules)]
        C1[AI Chatbot Service\n(Text)]
        C2[AI Voice Agent Service\n(Voice)]
        C3[Agent Assist Service\n(Real-time Guidance + Summaries)]
    end

    %% Agent & Workforce
    subgraph Agent_Workforce[Agent & Workforce Layer]
        D1[Unified Agent Desktop\n(CTI + CRM View)]
        D2[Workforce Management\n(WFM & QA)]
    end

    %% Integration & Core Systems
    subgraph Integration_Core[Integration & Core Systems]
        E0[API Gateway /\nIntegration Hub]
        E1[Claims & Eligibility Systems]
        E2[Billing & Payments]
        E3[EMR / EHR & RCM]
        E4[CRM / Case Management]
    end

    %% Data, Analytics & Governance
    subgraph Data_Analytics[Data, Analytics & Governance]
        F1[Event Streaming Bus]
        F2[Data Lake / Warehouse]
        F3[Analytics & Reporting\n(AHT, FCR, Containment, Deflection)]
        F4[Security, Compliance & Audit\n(HIPAA, IAM, Logging)]
    end

    %% Flows from channels into platform
    A1 --> B1
    A2 --> B2
    A3 --> B2
    A4 --> B2

    %% Routing into orchestrator
    B1 --> C0
    B2 --> C0

    %% Orchestrator to AI capabilities
    C0 -->|Simple Question| C1
    C0 -->|Transactional Task\n(PA, Rx, Status)| C2
    C0 -->|Complex / High Empathy| D1
    C0 -->|On-call Verification\n& Guidance| C3

    %% Agent assist feeds agent desktop
    C3 --> D1

    %% Orchestrator to integration hub and backends
    C1 --> E0
    C2 --> E0
    D1 --> E0

    E0 --> E1
    E0 --> E2
    E0 --> E3
    E0 --> E4

    %% Data & analytics wiring
    B1 -. events .-> F1
    C1 -. bot logs .-> F1
    C2 -. voice logs .-> F1
    C3 -. transcripts & KPIs .-> F1
    D1 -. interaction data .-> F1
    E0 -. API metrics .-> F1

    F1 --> F2
    F2 --> F3
    F3 --> D2

    %% Governance
    F4 --- B1
    F4 --- C1
    F4 --- C2
    F4 --- C3
    F4 --- E0
