import streamlit as st
import tempfile
import os,re
from dotenv import load_dotenv
from io import BytesIO
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Import the core logic functions
#from utils import  parse_java_zip,create_documents, setup_rag_pipeline, \
#    generate_documentation, generate_springboot_plan, generate_springboot_code_segmented

from rag_pipeline import   setup_rag_pipeline

from util.utility import parse_java_zip, parse_and_save_springboot_code, extract_file_list_from_blueprint
from docgen.java_docgen import create_documents, generate_documentation, evaluate_documentation, refine_documentation
from codegen.spring_boot_appgen import generate_springboot_plan, generate_springboot_code_segmented


# Load environment variables from .env file (if not set in the shell)
load_dotenv()

# Define the file separator used by the LLM
FILE_SEPARATOR = "--- FILE_START:"


# --- Streamlit UI Setup ---
st.set_page_config(
    page_title="Java Code Documenter & Spring Boot Application Generator",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.header("👨‍💻 AI-Powered Legacy Java to Spring Boot Migration Tool")
st.markdown(
    "Upload a Java codebase in a ZIP file to automatically generate detailed documentation and build a Spring Boot application that leverages LangChain, ChromaDB, Hugging Face embeddings, and Azure OpenAI.")
st.markdown("---")

# --- 1. File Uploader and Processing ---
st.header("1. Upload Java Codebase (ZIP) and Create RAG chain  ")
uploaded_file = st.file_uploader(
    "1. Upload Java Codebase (ZIP)",
    type="zip"
)

if uploaded_file:
    # Read the file content into a BytesIO object
    zip_bytes = uploaded_file.read()
    zip_io = BytesIO(zip_bytes)

    st.success(f"File uploaded: **{uploaded_file.name}**")


    # Cache the heavy processing steps
    @st.cache_resource(show_spinner="Analyzing and Vectorizing Java Code...")
    def process_and_setup(zip_file_io):
        """Processes zip, creates documents, and sets up the RAG chain."""
        # 1. Parse Java Files
        st.info("Step 1: Parsing Java files from ZIP...")
        java_files = parse_java_zip(zip_file_io)
        st.session_state.parsed_data = java_files

        if not java_files:
            st.error("No `.java` files found in the zip archive.")
            return None, None

        st.success(f"Found {len(java_files)} `java` files.")

        # 2. Create LangChain Documents
        documents = create_documents(java_files)

        # 3. Setup RAG Pipeline (Vector Store and Chain)
        st.info("Step 2: Creating vector store and RAG pipeline with ChromaDB...")
        vectorstore, qa_chain = setup_rag_pipeline(documents)

        st.success("RAG pipeline setup complete. Ready for generation.")
        return qa_chain


    qa_chain = process_and_setup(zip_io)
    st.session_state.step = 2


    if qa_chain:
        # --- 2. Documentation Generation Button ---

        st.header("2. Generate Comprehensive Documentation")

        if st.button("🚀 Generate Documentation", key="doc_button", type="primary"):

            with st.spinner("Generating documentation... This may take a moment."):
                # Generate Documentation
                documentation = generate_documentation(qa_chain)
                st.session_state.documentation = documentation

                st.subheader("✅ Documentation Generated")
                st.code(documentation, language="markdown")
                #st.markdown(documentation)

                st.download_button(
                    "Download as Markdown",
                    data=documentation,
                    file_name="documentation.md",
                    mime="text/markdown"
                )
                st.session_state.step = 3
        # --- 3. Documentation refinement Button ---
        if 'documentation' in st.session_state and st.session_state.documentation:
            st.header("3: Automated Documentation Evaluation & Refinement")

            st.write("Automatically evaluate documentation quality and identify areas needing manual review.")
            if 'evaluation' not in st.session_state:
                st.session_state.evaluation = None
            if 'refined_documentation' not in st.session_state:
                st.session_state.refined_documentation = None

            if st.session_state.evaluation is None:
                if st.button("Evaluate Documentation Quality", type="primary"):
                    with st.spinner("Evaluating documentation..."):
                        try:

                            evaluation = evaluate_documentation(
                                st.session_state.documentation
                            )
                            st.session_state.evaluation = evaluation
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error evaluating documentation: {str(e)}")

            if st.session_state.evaluation:
                evaluation = st.session_state.evaluation

                st.subheader("📊 Evaluation Dashboard")

                scores = {
                    'Completeness': evaluation.get('completeness_score', 0),
                    'Accuracy': evaluation.get('accuracy_score', 0),
                    'Clarity': evaluation.get('clarity_score', 0),
                    'Actionability': evaluation.get('actionability_score', 0),
                    'Detail Level': evaluation.get('detail_score', 0)
                }

                col1, col2 = st.columns([2, 1])

                with col1:
                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(scores.values()),
                            y=list(scores.keys()),
                            orientation='h',
                            marker=dict(
                                color=list(scores.values()),
                                colorscale='RdYlGn',
                                cmin=0,
                                cmax=100
                            ),
                            text=[f"{v}%" for v in scores.values()],
                            textposition='auto',
                        )
                    ])

                    fig.update_layout(
                        title="Documentation Quality Scores",
                        xaxis_title="Score",
                        yaxis_title="Criteria",
                        height=400,
                        xaxis=dict(range=[0, 100])
                    )

                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    overall_score = evaluation.get('overall_score', 0)

                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=overall_score,
                        title={'text': "Overall Score"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 75], 'color': "yellow"},
                                {'range': [75, 100], 'color': "lightgreen"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))

                    fig_gauge.update_layout(height=300)
                    st.plotly_chart(fig_gauge, use_container_width=True)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.subheader("⚠️ Missing Areas")
                    missing = evaluation.get('missing_areas', [])
                    if missing:
                        for area in missing:
                            st.warning(area)
                    else:
                        st.success("No missing areas identified")

                with col2:
                    st.subheader("🔍 Manual Review Needed")
                    manual_review = evaluation.get('manual_review_needed', [])
                    if manual_review:
                        for item in manual_review:
                            st.info(item)
                    else:
                        st.success("No manual review needed")

                with col3:
                    st.subheader("💡 Improvement Suggestions")
                    suggestions = evaluation.get('improvement_suggestions', [])
                    if suggestions:
                        for suggestion in suggestions:
                            st.info(suggestion)
                    else:
                        st.success("No suggestions")

                df_evaluation = pd.DataFrame([
                    {"Area": "Missing Areas", "Items": len(evaluation.get('missing_areas', []))},
                    {"Area": "Manual Review", "Items": len(evaluation.get('manual_review_needed', []))},
                    {"Area": "Improvements", "Items": len(evaluation.get('improvement_suggestions', []))}
                ])

                st.subheader("Summary Statistics")
                fig_pie = px.pie(df_evaluation, values='Items', names='Area', title='Review Areas Distribution')
                st.plotly_chart(fig_pie, use_container_width=True)

                st.divider()

                if st.session_state.refined_documentation is None:
                    if st.button("Refine Documentation", type="primary"):
                        st.session_state.step = 3
                        with st.spinner("Refining documentation based on evaluation..."):
                            try:
                                refined = refine_documentation(
                                    st.session_state.documentation,
                                    evaluation
                                )
                                st.session_state.refined_documentation = refined
                                st.success("Documentation refined successfully!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error refining documentation: {str(e)}")

                if st.session_state.refined_documentation:
                    st.subheader("Refined Documentation")
                    st.markdown(st.session_state.refined_documentation)

                    st.download_button(
                        label="Download Refined Documentation",
                        data=st.session_state.refined_documentation,
                        file_name="refined_documentation.md",
                        mime="text/markdown"
                    )
                    st.session_state.step = 4
        # --- 3. Spring Boot Generation (Conditional) ---
        if 'documentation' in st.session_state and st.session_state.refined_documentation:
            st.header("4. Generate Spring Boot Plan")

            if st.button("🏗️ Generate Spring Boot Blueprint", key="spring_button",type="primary"):

                with st.spinner("Analyzing Refined documentation to generate Spring Boot plan..."):
                    # Generate Spring Boot Plan
                    spring_plan = generate_springboot_plan(st.session_state.refined_documentation)

                    st.session_state.spring_plan = spring_plan

                    file_list = extract_file_list_from_blueprint(spring_plan)
                    st.session_state.target_file_list = file_list

                    st.subheader("✅ Spring Boot Application Blueprint")
                    st.code(spring_plan, language="markdown")

            if 'spring_plan' in st.session_state and st.session_state.spring_plan:
                # Display the Spring Boot Plan if already generated
                st.subheader("✅ Spring Boot Application Blueprint (Pre-Generated)")
                st.code(st.session_state.spring_plan, language="markdown")
                st.session_state.step = 5

        # Display the generated documentation if it exists in session state
        #elif 'documentation' in st.session_state and st.session_state.documentation:
        #    st.header("Generated Documentation")
        #    st.code(st.session_state.documentation, language="markdown")


        # --- 4. Spring Boot Generation (CODE) ---
        if 'documentation' in st.session_state and st.session_state.refined_documentation:
            st.header("5. Code Generation: Full Spring Boot Project 🏗️")


            if st.button("🚀 Generate & Save Spring Boot Code", key="code_gen_button", type="primary"):
                with st.spinner("Generating multi-file Spring Boot code... (pre-defined file list)"):
                    # 1. Generate the segmented code dictionary
                    # Call the new segmented function
                    generated_files_dict = generate_springboot_code_segmented(st.session_state.documentation,st.session_state.target_file_list)

                    if not generated_files_dict:
                        st.error("Code generation failed during the segmented process. See logs above.")
                        if 'temp_project_path' in st.session_state: del st.session_state.temp_project_path
                    else:
                        st.session_state.generated_files_dict = generated_files_dict

                        # 2. Save the files to a temporary directory
                        temp_project_path = parse_and_save_springboot_code(st.session_state.generated_files_dict)
                        st.session_state.temp_project_path = temp_project_path

                        st.success("✅ Spring Boot Project Code Generated and Saved!")

            if 'temp_project_path' in st.session_state and st.session_state.temp_project_path and st.session_state.temp_project_path != "Generation Failed.":
                project_path = st.session_state.temp_project_path

                st.subheader(f"Project Location: `{project_path}`")
                st.info("The generated files are saved temporarily. Review the structure below.")

                # ... (Display file structure logic remains the same) ...
                st.markdown("### Generated Project Structure")

                tree_output = []
                for root, dirs, files in os.walk(project_path):
                    level = root.replace(project_path, '').count(os.sep)
                    indent = ' ' * 4 * (level)
                    tree_output.append(f'{indent}📂 {os.path.basename(root)}/')
                    sub_indent = ' ' * 4 * (level + 1)
                    for f in files:
                        tree_output.append(f'{sub_indent}📄 {f}')

                st.code('\n'.join(tree_output), language='text')

                # Provide a simple way to view a sample file (e.g., the main application file)
                st.markdown("### Sample File Content (e.g., pom.xml)")
                try:
                    with open(os.path.join(project_path, 'pom.xml'), 'r') as f:
                        st.code(f.read(), language='xml')
                except FileNotFoundError:
                    try:
                        project_path += r"\\src\\main\\resources\\"
                        with open(os.path.join(project_path, 'pom.xml'), 'r') as f:
                            st.code(f.read(), language='xml')
                    except FileNotFoundError:
                        st.warning("Could not find pom.xml to display. Check the generated structure above.")

                if st.session_state.temp_project_path:
                    st.success("🎉 Migration Complete!")
                    st.balloons()


def render_sidebar():
    with st.sidebar:
        st.title("☕ Migration Workflow")

        steps = [
            "1. Upload Java Code",
            "2. Generate Documentation",
            "3. Evaluate & Refine",
            "4. Create Blueprint",
            "5. Generate Code"
        ]

        for i, step in enumerate(steps, 1):
            if i == st.session_state.step:
                st.markdown(f"**➡️ {step}**")
            elif i < st.session_state.step:
                st.markdown(f"✅ {step}")
            else:
                st.markdown(f"⬜ {step}")


        st.divider()

        if st.button("Reset Application", type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


def init_session_state():
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'parser' not in st.session_state:
        st.session_state.parser = None
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'documentation' not in st.session_state:
        st.session_state.documentation = None
    if 'evaluation' not in st.session_state:
        st.session_state.evaluation = None
    if 'refined_documentation' not in st.session_state:
        st.session_state.refined_documentation = None
    if 'blueprint' not in st.session_state:
        st.session_state.blueprint = None
    if 'generated_files' not in st.session_state:
        st.session_state.generated_files = {}
    if 'test_files' not in st.session_state:
        st.session_state.test_files = {}

def main():
    init_session_state()
    render_sidebar()

    #st.title("☕ Legacy Java to Spring Boot Migration Tool")
    st.markdown("---")


if __name__ == "__main__":
    main()

