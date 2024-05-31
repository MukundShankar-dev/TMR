import pandas as pd

df = pd.read_csv(f"old_outputs/tmr_cos_loss_0.15/contrastive_metrics_2/t2m_normal_keyid_metrics.csv")
print(df.head(25))