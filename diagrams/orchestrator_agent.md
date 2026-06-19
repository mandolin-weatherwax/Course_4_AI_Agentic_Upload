flowchart LR
    Orchestrator_Agent -->|1| Quoting_Agent
    Orchestrator_Agent -->|2| Inventory_Tool
    Orchestrator_Agent -->|3| Pricing_Review_Agent
    Orchestrator_Agent -->|4| Pricing_Tool
    Orchestrator_Agent -->|5| Order_Recommendation_Agent
    Orchestrator_Agent -->|6| Post_Stock_Transactions
    Orchestrator_Agent -->|7| Post_Sales_Transactions

    Post_Stock_Transactions --> db[(SQLite ledger)]
    Post_Sales_Transactions --> db