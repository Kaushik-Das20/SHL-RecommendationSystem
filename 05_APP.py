import streamlit as st
import requests
import pandas as pd

st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🎯",
    layout="wide"
)

API_URL = "http://localhost:8000"

st.title("SHL Assessment Recommendation System")
st.markdown("""
**Welcome!** This tool helps HR managers and recruiters find the right 
SHL assessments for their job roles instantly using AI.

Simply type your job description or query below and get instant recommendations!
""")

st.divider()


st.subheader(" Enter Your Query")

input_type = st.radio(
    "Choose input type:",
    ["Natural Language Query", "Job Description Text", "Job Description URL"],
    horizontal=True
)

if input_type == "Natural Language Query":
    user_input = st.text_area(
        "Type your query here:",
        placeholder="Example: I need a Java developer who is good at collaborating with business teams",
        height=100
    )

elif input_type == "Job Description Text":
    user_input = st.text_area(
        "Paste your Job Description here:",
        placeholder="Paste the full job description text here...",
        height=200
    )

else: 
    user_input = st.text_input(
        "Enter Job Description URL:",
        placeholder="https://example.com/job-description"
    )

num_results = st.slider(
    "Number of recommendations (5-10):",
    min_value=5,
    max_value=10,
    value=7
)

submit = st.button("🔍 Get Recommendations", type="primary", use_container_width=True)

st.divider()
st.subheader("Or Try a Sample Query")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Java Developer + Collaboration", use_container_width=True):
        user_input = "I am hiring for Java developers who can also collaborate effectively with my business teams"
        submit = True

with col2:
    if st.button("Python + SQL + JavaScript", use_container_width=True):
        user_input = "Looking to hire mid-level professionals proficient in Python, SQL and JavaScript"
        submit = True

with col3:
    if st.button("Analyst + Cognitive + Personality", use_container_width=True):
        user_input = "Hiring an analyst, want to screen using Cognitive and personality tests"
        submit = True

if submit and user_input:

    with st.spinner("AI is analyzing your query and finding best assessments..."):
        try:
            response = requests.post(
                f"{API_URL}/recommend",
                json={
                    "query": user_input,
                    "top_k": num_results
                },
                timeout=60
            )

            if response.status_code == 200:
                data = response.json()
                assessments = data["recommended_assessments"]

                st.divider()
                st.subheader(f"Top {data['total_results']} Recommended Assessments")
                st.caption(f"Query: *{data['query'][:100]}...*")

                total = data["total_results"]
                test_types = [a["test_type"] for a in assessments]
                remote_count = sum(1 for a in assessments if a["remote_testing"] == "Yes")

                m1, m2, m3 = st.columns(3)
                m1.metric("Total Recommendations", total)
                m2.metric("Remote Testing Available", f"{remote_count}/{total}")
                m3.metric("Test Types Covered", len(set(" ".join(test_types).split())))

                st.divider()

                st.subheader("📊 Results Table")

                table_data = []
                for a in assessments:
                    table_data.append({
                        "Assessment Name": a["assessment_name"],
                        "Test Type": a["test_type"],
                        "Remote Testing": a["remote_testing"],
                        "Adaptive/IRT": a["adaptive_irt"],
                        "URL": a["url"]
                    })

                df_display = pd.DataFrame(table_data)

                def make_clickable(url):
                    return f'<a href="{url}" target="_blank">View Assessment</a>'

                df_html = df_display.copy()
                df_html["URL"] = df_html["URL"].apply(make_clickable)
                st.write(df_html.to_html(escape=False, index=False), unsafe_allow_html=True)

                st.divider()

                st.subheader("Detailed View")

                for i, a in enumerate(assessments):
                    with st.expander(f"{i+1}. {a['assessment_name']}"):
                        col_a, col_b = st.columns(2)

                        with col_a:
                            st.markdown(f"**Test Type:** `{a['test_type']}`")
                            st.markdown(f"**Remote Testing:** {a['remote_testing']}")
                            st.markdown(f"**Adaptive/IRT:** {a['adaptive_irt']}")

                        with col_b:
                            st.markdown(f"** URL:** [View on SHL]({a['url']})")

                        if a["description"]:
                            st.markdown(f"**Description:** {a['description']}")

                st.divider()
                st.subheader("Test Type Legend")
                legend_col1, legend_col2 = st.columns(2)

                with legend_col1:
                    st.markdown("""
                    - **A** — Ability & Aptitude
                    - **B** — Biodata & Situational Judgement
                    - **C** — Competencies
                    - **D** — Development & 360
                    """)

                with legend_col2:
                    st.markdown("""
                    - **E** — Assessment Exercises
                    - **K** — Knowledge & Skills
                    - **P** — Personality & Behavior
                    - **S** — Simulations
                    """)

            else:
                st.error(f"API Error: {response.status_code} - {response.text}")

        except requests.exceptions.ConnectionError:
            st.error(""" Cannot connect to API. """)

        except Exception as e:
            st.error(f"Something went wrong: {str(e)}")

elif submit and not user_input:
    st.warning("Please enter a query first!")

st.divider()
st.caption("Built by Kaushik Das | SHL Assessment Recommendation System | Powered by Gemini AI + FAISS")
