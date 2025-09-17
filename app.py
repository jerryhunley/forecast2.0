# --- Sidebar ---
with st.sidebar:
    st.logo("assets/logo.png", link="https://1nhealth.com")
    
    st.header("⚙️ Setup")
    st.info("Start here by uploading your data files.")

    st.warning("🔒 **Privacy Notice:** Do not upload files containing PII.", icon="⚠️")
    pii_checkbox = st.checkbox("I confirm my files do not contain PII.")

    if pii_checkbox:
        uploaded_referral_file = st.file_uploader("1. Upload Referral Data (CSV)", type=["csv"])
        uploaded_funnel_def_file = st.file_uploader("2. Upload Funnel Definition (CSV/TSV)", type=["csv", "tsv"])
    else:
        uploaded_referral_file = None
        uploaded_funnel_def_file = None

st.title("📊 Recruitment Forecasting Tool")
st.header("Home & Data Setup")

if uploaded_referral_file and uploaded_funnel_def_file:
    st.info("Files uploaded. Click the button below to process and load the data.")
    if st.button("Process Uploaded Data", type="primary"):
        with st.spinner("Parsing files and processing data..."):
            try:
                referral_bytes_data = uploaded_referral_file.getvalue()
                header_df = pd.read_csv(io.BytesIO(referral_bytes_data), nrows=0, low_memory=False)
                pii_cols = [c for c in ["notes", "first name", "last name", "name", "phone", "email"] if c in [str(h).lower().strip() for h in header_df.columns]]

                if pii_cols:
                    original_col_names = [col for col in header_df.columns if str(col).lower().strip() in pii_cols]
                    st.error(f"PII Detected in columns: {', '.join(original_col_names)}. Please remove them and re-upload.", icon="🚫")
                    st.stop()
                
                funnel_def, ordered_st, ts_map = parse_funnel_definition(uploaded_funnel_def_file)
                
                if funnel_def and ordered_st and ts_map:
                    raw_df = pd.read_csv(io.BytesIO(referral_bytes_data))
                    processed_data = preprocess_referral_data(raw_df, funnel_def, ordered_st, ts_map)

                    if processed_data is not None and not processed_data.empty:
                        st.session_state.funnel_definition = funnel_def
                        st.session_state.ordered_stages = ordered_st
                        st.session_state.ts_col_map = ts_map
                        st.session_state.referral_data_processed = processed_data
                        st.session_state.inter_stage_lags = calculate_overall_inter_stage_lags(processed_data, ordered_st, ts_map)
                        st.session_state.site_metrics_calculated = calculate_site_metrics(processed_data, ordered_st, ts_map)
                        st.session_state.data_processed_successfully = True
                        st.success("Data processed successfully!")
                        st.rerun()
                    else:
                        st.error("Data processing failed after preprocessing.")
                else:
                    st.error("Funnel definition parsing failed.")
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                st.exception(e)

if st.session_state.data_processed_successfully:
    st.success("Data is loaded and ready.")
    st.info("👈 Please select an analysis page from the sidebar to view the results.")
else:
    st.info("👋 **Welcome to the Recruitment Forecasting Tool!**")
    st.markdown("""
        1.  **Confirm No PII**: Check the box in the sidebar.
        2.  **Upload Your Data**: Use the file uploaders in the sidebar.
        3.  **Process Data**: Click the "Process Uploaded Data" button.
        4.  **Explore**: Navigate to the analysis pages.
    """)