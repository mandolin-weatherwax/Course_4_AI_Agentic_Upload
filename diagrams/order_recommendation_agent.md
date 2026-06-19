flowchart LR
    Order_Recommendation_Agent --> t1[tool_gather_company_context]
    Order_Recommendation_Agent --> t2[tool_extract_price_bands]
    Order_Recommendation_Agent --> t3[tool_evaluate_pricing_signals]
    Order_Recommendation_Agent --> t4[tool_build_transaction_batches]
    Order_Recommendation_Agent --> t5[tool_validate_order_recommendation]
