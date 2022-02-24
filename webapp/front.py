import streamlit as st
import requests


model_selection_mapping = {'Without impossible Answers': 'without_impossible',
                           'With Impossible Answers': 'with_impossible'}

model_selection = st.sidebar.radio('Choose a model', options=['Without impossible Answers', 'With Impossible Answers'])
st.write("""
# SQuAD
Ask me a question!
""")

context_text = """Oxygen is a chemical element with symbol O and atomic number 8. It is a member of the chalcogen group on the periodic table and is a highly reactive nonmetal and oxidizing agent that readily forms compounds (notably oxides) with most elements. By mass, oxygen is the third-most abundant element in the universe, after hydrogen and helium. At standard temperature and pressure, two atoms of the element bind to form dioxygen, a colorless and odorless diatomic gas with the formula O
2. Diatomic oxygen gas constitutes 20.8% of the Earth's atmosphere. However, monitoring of atmospheric oxygen levels show a global downward trend, because of fossil-fuel burning. Oxygen is the most abundant element by mass in the Earth's crust as part of oxide compounds such as silicon dioxide, making up almost half of the crust's mass."""

question_text = "The atomic number of the periodic table for oxygen?"

with st.form(key='my_form'):
    context = st.text_area("Context!", value=context_text, height=200)
    question = st.text_area("Question!", value=question_text, height=2)
    submit_button = st.form_submit_button(label='Get the Answer')

if submit_button:
    if context == '' and question == '':
        st.error("You should provide context input and a question")
    elif context == '':
        st.error("Context cannot be empty")
    elif question == '':
        st.error("Question cannot be empty")
    else:
        answer = requests.post("http://0.0.0.0:1111/get-answer", json={'context': context, 'question': question,
                                                                        'model_to_use': model_selection_mapping[model_selection]})
        if answer.status_code == 200:
            ans = answer.json()["answer"]
            st.write(f'Answer: **{ans[0]}**')
        else:
            st.error('Backend error. Please try again.')
