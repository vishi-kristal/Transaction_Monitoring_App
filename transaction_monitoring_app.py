import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(layout="wide")
st.title("Transaction Monitoring Dashboard")

# -------- File Uploads -------- #
st.sidebar.header(" Upload Required Excel Files")
client_list_file = st.sidebar.file_uploader("Upload: Client_List.xlsx", type="xlsx")
user_aml_rating_file = st.sidebar.file_uploader("Upload: UserAmlRatingReport.xlsx", type="xlsx")
fund_deposits_file = st.sidebar.file_uploader("Upload: FundDeposits.xlsx", type="xlsx")
fund_withdrawals_file = st.sidebar.file_uploader("Upload: FundWithdrawals.xlsx", type="xlsx")
client_info_file = st.sidebar.file_uploader("Upload: ClientInfoReport.xlsx", type="xlsx")
exceptional_approval_file = st.sidebar.file_uploader("Upload: Exceptional Approval .xlsx", type="xlsx")
user_account_nav_file = st.sidebar.file_uploader("Upload: UserAccountNav.xlsx", type="xlsx")

if not all([client_list_file, user_aml_rating_file, fund_deposits_file, fund_withdrawals_file,
            client_info_file, exceptional_approval_file, user_account_nav_file]):
    st.warning(" Please upload all required Excel files to proceed.")
    st.stop()

# -------- Cached Master DF Builder -------- #
@st.cache_data
def build_master_df(client_list_file, user_aml_rating_file, fund_deposits_file, fund_withdrawals_file,
                    client_info_file, exceptional_approval_file, user_account_nav_file):
    fx_rates = {'USD': 1.30, 'SGD': 1.0, 'HKD': 0.17, 'CAD': 0.93, 'EUR': 1.46, 'AUD': 0.84, 'GBP': 1.73}

    client_list = pd.read_excel(client_list_file)
    user_aml_rating_report = pd.read_excel(user_aml_rating_file)
    fund_deposits = pd.read_excel(fund_deposits_file)
    fund_withdrawals = pd.read_excel(fund_withdrawals_file)
    client_info_report = pd.read_excel(client_info_file)
    exceptional_approval = pd.read_excel(exceptional_approval_file, sheet_name="KASG ", header=1)
    user_account_nav = pd.read_excel(user_account_nav_file)

    master_df = (
        client_list
        .merge(user_aml_rating_report, left_on='client_id', right_on='Kristal Client ID', how='left')
        .merge(client_info_report, on='client_id', how='left')
        .merge(fund_withdrawals.drop_duplicates(subset='client_id'), on='client_id', how='left')
    )
    master_df = master_df[['client_id', 'country_of_onboarding', 'country_of_residence_y', 'billing_type_y',
                           'kyc_status_y', 'Assigned to entity', 'Rating', 'Reason', 'quantum_of_wealth', 'location_of_bank_account']]
    master_df.columns = ['client_id', 'country_of_onboarding', 'country_of_residence', 'billing_type',
                         'kyc_status', 'assigned_to_entity', 'rating', 'reason', 'quantum_of_wealth', 'bank_location']

    def calculate_qow(quantum_of_wealth):
        mapping = {
            'LESS_THAN_0_5M': 250000, 'BETWEEN_0_5M_TO_1M': 750000, 'BETWEEN_1M_TO_2M': 1000000,
            'BETWEEN_2M_TO_5M': 2500000, 'MORE_THAN_5M': 7500000, 'LESS_THAN_10LAKHS': 8000,
            'BETWEEN_10LAKHS_TO_25LAKHS': 20000, 'BETWEEN_25LAKHS_TO_50LAKHS': 60000,
            'BETWEEN_50LAKHS_TO_1CR': 120000, 'MORE_THAN_1CR': 100000
        }
        return mapping.get(quantum_of_wealth, np.nan)

    master_df['qow'] = master_df['quantum_of_wealth'].apply(calculate_qow)

    fund_deposits = fund_deposits[fund_deposits['status_text'] == 'COMPLETED'].copy()
    fund_deposits['requested_time'] = pd.to_datetime(fund_deposits['requested_time'].astype(str).str[:10], errors='coerce')
    fund_deposits['requested_amount_SGD'] = fund_deposits.apply(lambda row: row['requested_amount'] * fx_rates.get(row['currency'], 1), axis=1)

    fund_withdrawals = fund_withdrawals[fund_withdrawals['internal_status'] == 'COMPLETED'].copy()
    fund_withdrawals['request_time'] = pd.to_datetime(fund_withdrawals['request_time'].astype(str).str[:10], errors='coerce')
    fund_withdrawals['requested_amount_SGD'] = fund_withdrawals.apply(lambda row: row['request_amount'] * fx_rates.get(row['request_currency'], 1), axis=1)

    latest_month = max(fund_deposits['requested_time'].max(), fund_withdrawals['request_time'].max()).to_period('M')
    start_month = latest_month - 11

    fund_deposits = fund_deposits[fund_deposits['requested_time'].dt.to_period('M').between(start_month, latest_month)]
    fund_withdrawals = fund_withdrawals[fund_withdrawals['request_time'].dt.to_period('M').between(start_month, latest_month)]

    def calculate_monthly_data(df, date_col, amount_col, client_id_col, prefix):
        df['month'] = df[date_col].dt.to_period('M')
        monthly_counts = df.groupby([client_id_col, 'month']).size().unstack(fill_value=0)
        monthly_amounts = df.groupby([client_id_col, 'month'])[amount_col].sum().unstack(fill_value=0)
        monthly_counts.columns = [f'{prefix}_count_T-{i}' for i in range(len(monthly_counts.columns))]
        monthly_amounts.columns = [f'{prefix}_amount_T-{i}' for i in range(len(monthly_amounts.columns))]
        return monthly_counts, monthly_amounts

    deposit_counts, deposit_amounts = calculate_monthly_data(fund_deposits, 'requested_time', 'requested_amount_SGD', 'client_id', 'deposit')
    withdrawal_counts, withdrawal_amounts = calculate_monthly_data(fund_withdrawals, 'request_time', 'requested_amount_SGD', 'client_id', 'withdrawal')

    master_df = master_df.merge(deposit_counts, on='client_id', how='left') \
                         .merge(deposit_amounts, on='client_id', how='left') \
                         .merge(withdrawal_counts, on='client_id', how='left') \
                         .merge(withdrawal_amounts, on='client_id', how='left')

    exceptional_approval.columns = exceptional_approval.columns.str.strip().str.lower().str.replace(" ", "_")
    exceptional_approval['approval_provided'] = exceptional_approval['approval_provided'].astype(str).str.strip().str.upper()
    exceptional_approval['exceptional_approval'] = exceptional_approval['approval_provided'].apply(lambda x: 'YES' if x == 'YES' else 'NO')
    approval_clean = exceptional_approval[['client_id', 'exceptional_approval']].drop_duplicates(subset='client_id')
    master_df = master_df.merge(approval_clean, on='client_id', how='left')
    master_df['exceptional_approval'] = master_df['exceptional_approval'].fillna('NO')

    user_account_nav = user_account_nav.groupby("client_id", as_index=False)["Cash Transferred in USD"].sum()
    user_account_nav["Cash Transferred in SGD"] = user_account_nav["Cash Transferred in USD"] * fx_rates["USD"]
    master_df = master_df.merge(user_account_nav[["client_id", "Cash Transferred in SGD"]], on="client_id", how="left")

    master_df.fillna(0, inplace=True)
    return master_df

# Run the cached pipeline
master_df = build_master_df(client_list_file, user_aml_rating_file, fund_deposits_file, fund_withdrawals_file,
                            client_info_file, exceptional_approval_file, user_account_nav_file)
# Download full master data
st.download_button("⬇ Download Full Master Data", master_df.to_csv(index=False).encode("utf-8"), "full_master_data.csv", "text/csv")


# -------- Apply Rules -------- #
def apply_rules(row):
    reasons = []
    if row.get("rating") in ["HIGH", "VERY_HIGH"]: reasons.append("Rule 1")
    if any(row.get(f"deposit_count_T-{i}", 0) > 5 for i in range(12)): reasons.append("Rule 2")
    if any(row.get(f"withdrawal_count_T-{i}", 0) > 5 for i in range(12)): reasons.append("Rule 3")
    if any(row.get(f"deposit_amount_T-{i}", 0) > 300000 for i in range(12)): reasons.append("Rule 4")
    if any(row.get(f"withdrawal_amount_T-{i}", 0) > 300000 for i in range(12)): reasons.append("Rule 5")
    if row.get("qow", 0) > 0 and row.get("Cash Transferred in SGD", 0) > 0.6 * row.get("qow"): reasons.append("Rule 6")
    if isinstance(row.get("asset_transfer_date"), pd.Timestamp) and isinstance(row.get("withdrawal_date"), pd.Timestamp):
        if 0 < (row["withdrawal_date"] - row["asset_transfer_date"]).days <= 180: reasons.append("Rule 7")
    if row.get("bank_location") != row.get("country_of_residence"): reasons.append("Rule 8")

    n_rules = len(reasons)
    label = "Low Risk" if n_rules == 0 else "Moderate Risk" if n_rules <= 2 else "High Risk"
    return pd.Series([label, ", ".join(reasons), set(reasons)], index=["risk_label", "risk_reason", "rules_triggered_set"])

# -------- Rule Application -------- #
df = master_df.copy()
df = df.drop(columns=["risk_label", "risk_reason", "rules_triggered_set"], errors="ignore")
results = df.apply(apply_rules, axis=1)
df = pd.concat([df, results], axis=1)

# -------- Filters -------- #

st.sidebar.header("Filter by Rules")
all_rules = [f"Rule {i}" for i in range(1, 9)]
selected_rules = st.sidebar.multiselect("Show clients triggering any of these rules:", all_rules)

st.sidebar.header("Exceptional Approval Filter")
approval_filter = st.sidebar.selectbox("Show clients with Exceptional Approval:", ["All", "YES", "NO"])



# Apply filters
filtered_df = df.copy()  # Reinitialize before applying selected filters
if selected_rules:
    filtered_df = filtered_df[filtered_df["rules_triggered_set"].apply(lambda x: bool(x & set(selected_rules)))]
if approval_filter != "All":
    filtered_df = filtered_df[filtered_df["exceptional_approval"] == approval_filter]
# -------- Display Sections -------- #
st.subheader(" Clients per Rule Violation")
rule_descriptions = {
    f"Rule {i}": desc for i, desc in enumerate([
        "Clients who are High or Very High Risk based on country of residence, citizenship, Tax residence or world check",
        "More than 5 deposits in any month in last 12 months",
        "More than 5 withdrawals in any month in last 12 months",
        "Deposits > 300k SGD in a month",
        "Withdrawals > 300k SGD in a month",
        "Net cash > 60% of QOW",
        "Withdrawal < 6 months after asset transfer",
        "Bank location ≠ country of residence"], 1)}
rule_data = [{"# Rule": rule, "Description": rule_descriptions[rule], "# of clients": df["risk_reason"].str.contains(rule).sum()} for rule in all_rules]
st.dataframe(pd.DataFrame(rule_data))

st.subheader(" Risk Category Counts")
st.write("Low Risk: No rules triggered<br>Moderate Risk: 1-2 rules triggered<br>High Risk: 3 or more rules triggered", unsafe_allow_html=True)
risk_counts = (
    filtered_df["risk_label"].value_counts()
    .rename_axis("risk_label")
    .reindex(["Low Risk", "Moderate Risk", "High Risk"], fill_value=0)
    .reset_index(name="count")
)

st.dataframe(risk_counts, use_container_width=True)

st.subheader(" Flagged Clients")
st.write("T0: July, 2024, T1: August, 2024, T2: September, 2024")
filtered_df["client_id"] = filtered_df["client_id"].astype(str)
st.dataframe(filtered_df)

csv = filtered_df.drop(columns=["rules_triggered_set"], errors="ignore").to_csv(index=False).encode("utf-8")
st.download_button("⬇ Download Filtered Report", csv, "filtered_risk_report.csv", "text/csv")

