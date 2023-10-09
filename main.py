import streamlit as st
import preprocess_text
import tensorflow as tf
import loaded_model
# Title and description
st.title("SkimLit")
st.write("Enter Abstract below:")

# Text area for user input
paragraph = st.text_area("Enter your Abstract here", height=200)

# paragraph = "The recently released GPT-4 Code Interpreter has demonstrated remarkable proficiency in solving challenging math problems, primarily attributed to its ability to seamlessly reason with natural language, generate code, execute code, and continue reasoning based on the execution output. In this paper, we present a method to fine-tune open-source language models, enabling them to use code for modeling and deriving math equations and, consequently, enhancing their mathematical reasoning abilities. We propose a method of generating novel and high-quality datasets with math problems and their code-based solutions, referred to as MathCodeInstruct. Each solution interleaves natural language, code, and execution results. We also introduce a customized supervised fine-tuning and inference approach. This approach yields the MathCoder models, a family of models capable of generating code-based solutions for solving challenging math problems. Impressively, the MathCoder models achieve state-of-the-art scores among open-source LLMs on the MATH (45.2%) and GSM8K (83.9%) datasets, substantially outperforming other open-source alternatives. Notably, the MathCoder model not only surpasses ChatGPT-3.5 and PaLM-2 on GSM8K and MATH but also outperforms GPT-4 on the competition-level MATH dataset."

# Create a Streamlit button
process_button = st.button("Process Abstract....")
if process_button:
    df, test_sentences = preprocess_text.split_paragraph_to_dataframe(paragraph)

    test_chars = [preprocess_text.split_chars(sentence) for sentence in test_sentences]

    test_line_numbers_one_hot, test_total_lines_one_hot = preprocess_text.one_hot(df,tf)

    x = (test_line_numbers_one_hot,
         test_total_lines_one_hot,
         tf.constant(test_sentences),
         tf.constant(test_chars))

    predicted_class = loaded_model.loaded_model(x,tf)
    df['Predicted class'] = predicted_class
    df.drop(['Line number','Total lines'], axis=1,inplace=True)
    st.dataframe(df)


# test_pos_char_token_labels = tf.data.Dataset.from_tensor_slices(test_labels_one_hot)
# test_pos_char_token_dataset = tf.data.Dataset.zip((test_pos_char_token_data, test_pos_char_token_labels))
##test_pos_char_token_dataset = test_pos_char_token_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
# Display the input paragraph
# st.header("Output :")
# st.write(user_input)
