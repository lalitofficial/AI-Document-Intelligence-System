import requests
import streamlit as st

DEFAULT_API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(page_title="IDFC GenAI - Processing Demo", layout="centered")
st.title("IDFC GenAI - Processing Demo")
st.write("Upload an image or PDF and submit it to the processing API.")

with st.sidebar:
    st.header("API")
    api_base = st.text_input("Base URL", value=DEFAULT_API_BASE)
    api_base = api_base.rstrip("/")
    st.caption("Run the API with `python run.py` and this UI with `streamlit run streamlit_app.py`.")

submit_tab, status_tab, job_tab = st.tabs(["Submit", "Status", "Job Details"])

with submit_tab:
    with st.form("submit_form", clear_on_submit=False):
        email = st.text_input("Email", value="sourav@gmail.com")
        processor = st.text_input("Processor", value="online-something")
        uploaded_file = st.file_uploader(
            "Object file",
            type=["png", "jpg", "jpeg", "pdf"],
            help="Upload a file to process."
        )
        submitted = st.form_submit_button("Submit", type="primary", use_container_width=True)

    if submitted:
        if not uploaded_file:
            st.warning("Please choose a file before submitting.")
        else:
            try:
                with st.spinner("Submitting job..."):
                    response = requests.post(
                        f"{api_base}/submit",
                        data={"email": email, "processor": processor},
                        files={
                            "object_file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                uploaded_file.type or "application/octet-stream",
                            )
                        },
                        timeout=30,
                    )
                if response.ok:
                    st.success("Job queued.")
                    st.json(response.json())
                else:
                    st.error(f"Submit failed ({response.status_code}).")
                    st.code(response.text)
            except requests.RequestException as exc:
                st.error(f"Request failed: {exc}")

with status_tab:
    with st.form("status_form", clear_on_submit=False):
        status_email = st.text_input("Email to lookup", value="sourav@gmail.com")
        status_submit = st.form_submit_button("Fetch Status", use_container_width=True)

    if status_submit:
        try:
            with st.spinner("Fetching status..."):
                response = requests.get(f"{api_base}/status/{status_email}", timeout=15)
            if response.ok:
                st.json(response.json())
            else:
                st.error(f"Status lookup failed ({response.status_code}).")
                st.code(response.text)
        except requests.RequestException as exc:
            st.error(f"Request failed: {exc}")

with job_tab:
    with st.form("job_form", clear_on_submit=False):
        job_id = st.text_input("Job ID")
        job_submit = st.form_submit_button("Fetch Job", use_container_width=True)

    if job_submit:
        if not job_id.strip():
            st.warning("Enter a job ID to lookup.")
        else:
            try:
                with st.spinner("Fetching job details..."):
                    response = requests.get(f"{api_base}/job/{job_id.strip()}", timeout=15)
                if response.ok:
                    st.json(response.json())
                else:
                    st.error(f"Job lookup failed ({response.status_code}).")
                    st.code(response.text)
            except requests.RequestException as exc:
                st.error(f"Request failed: {exc}")
