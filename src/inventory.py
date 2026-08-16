import math

Z_95 = 1.645

def inventory_policy(mean_daily_demand, demand_std, lead_time_days=5, service_level_z=Z_95):
    safety_stock = service_level_z * demand_std * math.sqrt(lead_time_days)
    reorder_point = mean_daily_demand * lead_time_days + safety_stock
    return {
        "safety_stock": float(max(0, safety_stock)),
        "reorder_point": float(max(0, reorder_point)),
        "suggested_reorder_qty": float(max(0, reorder_point))
    }
