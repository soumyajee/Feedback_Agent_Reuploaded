import streamlit as st
import pandas as pd
import io
import tempfile
import os
import logging
from typing import Dict, List, Optional
import json
from collections import defaultdict
from statistics import mean, median

# Configure page FIRST - before any other Streamlit commands
st.set_page_config(
    page_title="AIME Assessment Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import your existing classes (you'll need to ensure these are available)
try:
    from agant_updated_assement import ProfileStorage, AssessmentItem, StudentProfile, TeacherProfile, ConfigurableLLM, get_config, AgentConfig
    from agant_updated_feedback_aggent import FeedbackAgent, FeedbackSummary
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    st.error("Required modules not found. Please ensure all dependencies are installed.")

def create_sample_csv():
    """Create a sample CSV for download"""
    sample_data = '''"student_id","teacher_id","objective","question","answer"
"student_001","teacher_001","Understand AI concepts","What is the primary role of ECU diagnostics in vehicles?","ECU diagnostics monitor vehicle systems to detect faults and ensure performance."
"student_002","teacher_002","Apply AI concepts","How do AUTOSAR protocols enhance ECU data processing?","AUTOSAR protocols enable consistent data exchange, improving ECU data processing efficiency."
"student_003","teacher_003","Analyze AI applications","What advantages does AI bring to AUTOSAR-based fault detection?","AI analyzes AUTOSAR data to predict faults with higher accuracy using real-time ECU inputs."'''
    return sample_data

def validate_csv_format(df):
    """Validate that the CSV has required columns"""
    required_columns = ['student_id', 'teacher_id', 'objective', 'question', 'answer']
    
    # Check if DataFrame is empty
    if df.empty:
        return False, "CSV file is empty"
    
    # Check for required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    # Remove any completely empty rows
    df_clean = df.dropna(how='all')
    
    # Check for empty values in required columns
    empty_rows = df_clean[required_columns].isnull().any(axis=1).sum()
    if empty_rows > 0:
        return False, f"Found {empty_rows} rows with empty values in required columns"
    
    # Check for extra columns that might indicate CSV parsing issues
    extra_columns = [col for col in df.columns if col not in required_columns and not col.startswith('Unnamed')]
    if extra_columns:
        st.warning(f"Found additional columns that will be ignored: {', '.join(extra_columns)}")
    
    return True, f"CSV format is valid. Found {len(df_clean)} valid records."

def display_student_summaries(feedback):
    """Display student summaries in a formatted way"""
    st.subheader("📚 Student Performance Summary")
    
    if not feedback.student_summaries:
        st.info("No student data available.")
        return
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Overview", "Detailed View"])
    
    with tab1:
        # Create a summary dataframe
        summary_data = []
        for student_id, data in feedback.student_summaries.items():
            summary_data.append({
                'Student ID': student_id,
                'Name': data['name'],
                'Average Score': f"{data['average_score']:.2f}",
                'Total Assessments': data['total_assessments'],
                'Trend': data['performance_trend']
            })
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True)
    
    with tab2:
        # Detailed view with expandable sections
        for student_id, data in feedback.student_summaries.items():
            with st.expander(f"👤 {data['name']} ({student_id})"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Average Score", f"{data['average_score']:.2f}")
                with col2:
                    st.metric("Total Assessments", data['total_assessments'])
                with col3:
                    st.metric("Trend", data['performance_trend'])
                
                # Objective breakdown
                if data['objective_breakdown']:
                    st.write("**Objective Performance:**")
                    obj_df = pd.DataFrame([
                        {'Objective': obj, 'Score': f"{score:.2f}"} 
                        for obj, score in data['objective_breakdown'].items()
                    ])
                    st.dataframe(obj_df, use_container_width=True)
                
                # Bloom taxonomy breakdown
                if data['bloom_breakdown']:
                    st.write("**Bloom Taxonomy Performance:**")
                    bloom_df = pd.DataFrame([
                        {'Bloom Level': bloom, 'Score': f"{score:.2f}"} 
                        for bloom, score in data['bloom_breakdown'].items()
                    ])
                    st.dataframe(bloom_df, use_container_width=True)
                
                # Individual assessments
                if data['assessments']:
                    st.write("**Individual Assessments:**")
                    for i, assessment in enumerate(data['assessments'], 1):
                        with st.container():
                            st.write(f"**Assessment {i}:**")
                            st.write(f"*Question:* {assessment['question']}")
                            st.write(f"*Answer:* {assessment['answer']}")
                            st.write(f"*Score:* {assessment['score']:.2f} | *Bloom Level:* {assessment['bloom_level']} | *Type:* {assessment['question_type']}")
                            st.divider()

def display_teacher_summaries(feedback):
    """Display teacher summaries"""
    st.subheader("👨‍🏫 Teacher Performance Summary")
    
    if not feedback.teacher_summaries:
        st.info("No teacher data available.")
        return
    
    for teacher_id, data in feedback.teacher_summaries.items():
        with st.expander(f"🏫 {data['teacher_name']} ({teacher_id})"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Average Score", f"{data['average_score']:.2f}")
                st.metric("Total Students", data['total_students'])
            
            with col2:
                st.write("**Score Distribution:**")
                dist = data['score_distribution']
                st.write(f"• Min: {dist['min']:.2f}")
                st.write(f"• Max: {dist['max']:.2f}")
                st.write(f"• Median: {dist['median']:.2f}")
                st.write(f"• Average: {dist['average']:.2f}")
            
            if data['top_students']:
                st.write(f"**Top Students:** {', '.join(data['top_students'])}")
            if data['low_students']:
                st.write(f"**Students Needing Support:** {', '.join(data['low_students'])}")
            
            st.info(data['summary'])

def display_objective_summaries(feedback):
    """Display objective summaries"""
    st.subheader("🎯 Objective Performance Summary")
    
    if not feedback.objective_summaries:
        st.info("No objective data available.")
        return
    
    # Create a summary table
    obj_data = []
    for objective, data in feedback.objective_summaries.items():
        obj_data.append({
            'Objective': objective,
            'Average Score': f"{data['average_score']:.2f}",
            'Total Questions': data['total_questions'],
            'Mastery %': f"{data['mastery_percentage']:.1f}%"
        })
    
    df_objectives = pd.DataFrame(obj_data)
    st.dataframe(df_objectives, use_container_width=True)
    
    # Detailed view
    for objective, data in feedback.objective_summaries.items():
        with st.expander(f"📋 {objective}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Average Score", f"{data['average_score']:.2f}")
            with col2:
                st.metric("Total Questions", data['total_questions'])
            with col3:
                st.metric("Mastery Percentage", f"{data['mastery_percentage']:.1f}%")
            
            st.info(data['summary'])

def display_learning_gaps(feedback):
    """Display learning gaps"""
    st.subheader("⚠️ Learning Gaps")
    
    if not feedback.learning_gaps:
        st.success("🎉 No significant learning gaps detected!")
        return
    
    for objective, gap_data in feedback.learning_gaps.items():
        st.warning(f"**{objective}**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"Average Score: {gap_data['average_score']:.2f}")
        with col2:
            st.write(f"Mastery Percentage: {gap_data['mastery_percentage']:.1f}%")
        
        st.write(f"💡 **Recommendation:** {gap_data['recommendation']}")
        st.divider()

def display_low_performing_students(feedback):
    """Display low-performing students"""
    st.subheader("📉 Students Needing Additional Support")
    
    if not feedback.low_performing_students:
        st.success("🎉 No low-performing students detected!")
        return
    
    for student in feedback.low_performing_students:
        st.warning(f"**{student['name']} ({student['student_id']})**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Average Score: {student['average_score']:.2f}")
        with col2:
            if student['weak_objectives']:
                st.write("**Weak Objectives:**")
                for obj, score in student['weak_objectives'].items():
                    st.write(f"• {obj}: {score:.2f}")
        
        if student['weak_bloom_levels']:
            st.write("**Weak Bloom Levels:**")
            for bloom, score in student['weak_bloom_levels'].items():
                st.write(f"• {bloom}: {score:.2f}")
        
        st.write(f"💡 **Recommendation:** {student['recommendation']}")
        st.divider()

def display_misunderstood_concepts(feedback):
    """Display misunderstood concepts"""
    st.subheader("🤔 Misunderstood Concepts")
    
    if not feedback.misunderstood_concepts:
        st.success("🎉 No misunderstood concepts detected!")
        return
    
    for concept_data in feedback.misunderstood_concepts:
        st.error(f"**{concept_data['concept']}**")
        st.write(f"**Question:** {concept_data['question']}")
        st.write(f"**Average Score:** {concept_data['average_score']:.2f}")
        st.write(f"💡 **Recommendation:** {concept_data['recommendation']}")
        st.divider()

def main():
    # Main title
    st.title("🤖 AIME Assessment Feedback Agent Dashboard")
    st.markdown("Upload your assessment CSV file to generate comprehensive feedback reports.")
    
    # Sidebar with logo at the top
    with st.sidebar:
        # Add AIME logo at the very top
        try:
            st.image("aime-logo-RGB (3).png", width=150)
        except FileNotFoundError:
            # Fallback if logo file is not found
            st.markdown("""
                <div style="text-align: center; padding: 1rem; margin-bottom: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
                    <h2 style="margin: 0; font-family: Arial, sans-serif;">🤖 AIME</h2>
                    <p style="margin: 0; font-size: 0.8rem;">Assessment Intelligence</p>
                </div>
            """, unsafe_allow_html=True)
            st.warning("Logo file 'aime-logo-RGB (3).png' not found. Using fallback design.")
        
        st.header("🛠️ Navigation")
        
        # File upload section
        st.subheader("📁 File Upload")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload a CSV file with columns: student_id, teacher_id, objective, question, answer"
        )
        
        # Sample CSV download
        st.subheader("📥 Sample CSV")
        sample_csv = create_sample_csv()
        st.download_button(
            label="Download Sample CSV",
            data=sample_csv,
            file_name="sample_assessment_data.csv",
            mime="text/csv"
        )
        
        # Menu selection
        if uploaded_file is not None:
            st.subheader("📋 Report Sections")
            menu_options = [
                "Overview",
                "Student Performance",
                "Teacher Performance", 
                "Objective Analysis",
                "Learning Gaps",
                "Low Performers",
                "Misunderstood Concepts"
            ]
            
            selected_sections = st.multiselect(
                "Select sections to display:",
                menu_options,
                default=["Overview"]
            )
        else:
            selected_sections = ["Overview"]
    
    # Main content area
    if not IMPORTS_AVAILABLE:
        st.error("Please ensure all required modules are properly installed.")
        return
    
    if uploaded_file is None:
        st.info("👆 Please upload a CSV file using the sidebar to get started.")
        
        # Display instructions
        st.subheader("📝 Instructions")
        st.write("""
        1. **Upload CSV**: Use the sidebar to upload your assessment data CSV file
        2. **Required Columns**: Your CSV must contain these columns:
           - `student_id`: Unique identifier for each student
           - `teacher_id`: Identifier for the teacher
           - `objective`: Learning objective being assessed
           - `question`: The assessment question
           - `answer`: Student's response
        3. **Select Reports**: Choose which sections of the feedback report to display
        4. **View Results**: Analyze the comprehensive feedback and recommendations
        """)
        
        # Show sample data format
        st.subheader("📋 Sample Data Format")
        sample_df = pd.read_csv(io.StringIO(create_sample_csv()))
        st.dataframe(sample_df, use_container_width=True)
        
    else:
        try:
            # Read and validate the uploaded file with proper CSV handling
            df = pd.read_csv(
                uploaded_file,
                encoding='utf-8',
                quotechar='"',
                escapechar='\\',
                skipinitialspace=True,
                on_bad_lines='warn'  # This will warn about bad lines but continue processing
            )
            
            # Validate CSV format
            is_valid, message = validate_csv_format(df)
            
            if not is_valid:
                st.error(f"❌ {message}")
                st.subheader("📋 Your Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                return
            
            st.success("✅ CSV format validated successfully!")
            
            # Display data preview
            with st.expander("👀 Data Preview", expanded=False):
                st.dataframe(df, use_container_width=True)
                st.write(f"**Total Records:** {len(df)}")
                st.write(f"**Unique Students:** {df['student_id'].nunique()}")
                st.write(f"**Unique Teachers:** {df['teacher_id'].nunique()}")
                st.write(f"**Unique Objectives:** {df['objective'].nunique()}")
            
            # Process the file
            with st.spinner("🔄 Processing assessment data..."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp_file:
                    df.to_csv(tmp_file.name, index=False)
                    tmp_path = tmp_file.name
                
                try:
                    # Initialize the feedback agent
                    config = get_config()
                    profile_storage = ProfileStorage(db_path=config.PROFILE_DB_PATH)
                    llm = ConfigurableLLM(
                        config=config,
                        api_key=config.GROQ_API_KEY,
                        model=config.LLM_MODEL,
                        base_url=config.LLM_BASE_URL,
                        max_tokens=config.MAX_TOKENS
                    )
                    
                    feedback_agent = FeedbackAgent(
                        config=config,
                        profile_storage=profile_storage,
                        llm=llm,
                        csv_path=tmp_path
                    )
                    
                    # Process the data
                    feedback_agent.read_csv()
                    feedback = feedback_agent.generate_feedback()
                    
                    st.success("✅ Analysis complete!")
                    
                    # Display selected sections
                    if "Overview" in selected_sections:
                        st.header("📈 Overview")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Students", len(feedback.student_summaries))
                        with col2:
                            st.metric("Total Teachers", len(feedback.teacher_summaries))
                        with col3:
                            st.metric("Learning Gaps", len(feedback.learning_gaps))
                        with col4:
                            st.metric("Low Performers", len(feedback.low_performing_students))
                    
                    if "Student Performance" in selected_sections:
                        display_student_summaries(feedback)
                    
                    if "Teacher Performance" in selected_sections:
                        display_teacher_summaries(feedback)
                    
                    if "Objective Analysis" in selected_sections:
                        display_objective_summaries(feedback)
                    
                    if "Learning Gaps" in selected_sections:
                        display_learning_gaps(feedback)
                    
                    if "Low Performers" in selected_sections:
                        display_low_performing_students(feedback)
                    
                    if "Misunderstood Concepts" in selected_sections:
                        display_misunderstood_concepts(feedback)
                    
                finally:
                    # Clean up temporary file
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                        
        except Exception as e:
            st.error(f"❌ An error occurred while processing the file: {str(e)}")
            logger.error(f"Error processing file: {e}")

if __name__ == "__main__":
    main()