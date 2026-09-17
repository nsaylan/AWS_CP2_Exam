import streamlit as st
from pypdf import PdfReader
import re
import os
import random
import chromadb
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------------
#  PDF PARSING FUNCTION
# ------------------------------------------------------------------
def parse_aws_questions(pdf_content, debug=True):
    """
    Reads the PDF line by line and parses questions.
    Two-stage approach: first collect all questions, then build final list.
    The final stage MERGES questions with the same number, keeping the
    most complete version (the one with the correct answer).
    """
    text = ""
    try:
        pdf_reader = PdfReader(pdf_content)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        st.error(f"PDF read error: {e}")
        return []

    # --- Pre-cleaning ---
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if re.match(r'^=+\s*Page\s+\d+', stripped):
            continue
        if re.match(r'^=+$', stripped):
            continue
        if 'shapingpixel.com' in stripped.lower():
            continue
        if stripped.startswith('[text layer]'):
            continue
        if stripped.startswith('#'):
            line = stripped.lstrip('#').strip()
            if line:
                cleaned_lines.append(line)
            continue
        cleaned_lines.append(line)

    # STAGE 1: Collect all questions (with or without answers)
    raw_questions = []
    current_question = None
    current_options = []
    current_correct = None
    answer_seen_for_current = False

    question_start_pattern = re.compile(r'^\s*(\d+)\s*[\)\.\-]\s*(.*)')
    # Daha esnek option pattern: "A. ", "A) ", "A - ", "A: " formatlarını yakalar
    option_pattern = re.compile(r'^([A-E])\s*[\)\.\-\:]\s*(.+)')
    # "A)" veya "A." gibi tek başına harf içeren satırlar (seçenek metni alt satırda)
    option_pattern_loose = re.compile(r'^([A-E])\s*[\)\.\-]\s*$')
    correct_answer_pattern = re.compile(
        r'(?:correct\s*answers?|answer\s*\(s\)|answer|ans)\s*[:\-]?\s*([A-E](?:\s*,?\s*[A-E]){0,4})\b',
        re.IGNORECASE
    )

    def save_current():
        nonlocal current_question, current_options, current_correct, answer_seen_for_current
        if current_question and len(current_options) >= 2:
            q_match = re.match(r'^(\d+)\)\s*(.*)', current_question, re.DOTALL)
            if q_match:
                q_num = q_match.group(1)
                q_text = q_match.group(2).strip()
            else:
                q_num = ""
                q_text = current_question.strip()

            raw_questions.append({
                'num': q_num,
                'text': q_text,
                'options': current_options.copy(),
                'correct': current_correct if current_correct else ""
            })
        elif current_question:
            # Tek seçenekli veya seçeneksiz soruları da kaydet (debug için)
            q_match = re.match(r'^(\d+)\)\s*(.*)', current_question, re.DOTALL)
            if q_match:
                q_num = q_match.group(1)
                q_text = q_match.group(2).strip()
            else:
                q_num = ""
                q_text = current_question.strip()
            raw_questions.append({
                'num': q_num,
                'text': q_text,
                'options': current_options.copy(),
                'correct': current_correct if current_correct else ""
            })
        current_question = None
        current_options = []
        current_correct = None
        answer_seen_for_current = False

    i = 0
    while i < len(cleaned_lines):
        line = cleaned_lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # Cevap görüldüyse, yeni soru başlayana kadar her şeyi atla
        if answer_seen_for_current:
            if question_start_pattern.match(stripped):
                pass
            else:
                i += 1
                continue

        # Cevap satırı mı?
        if current_question is not None:
            m = correct_answer_pattern.search(stripped)
            if m:
                candidate = m.group(1).strip().upper().replace(" ", "").replace(",", "")
                if re.match(r'^[A-E]{1,5}$', candidate):
                    if not option_pattern.match(stripped) and not option_pattern_loose.match(stripped):
                        current_correct = candidate
                        answer_seen_for_current = True
                        i += 1
                        continue

        # Yeni soru başlangıcı mı?
        q_match = question_start_pattern.match(stripped)
        if q_match:
            q_num = q_match.group(1)
            q_text = q_match.group(2).strip()
            save_current()
            current_question = f"{q_num}) {q_text}" if q_text else f"{q_num})"
            current_options = []
            current_correct = None
            answer_seen_for_current = False
            i += 1
            continue

        # Seçenek satırı mı?
        if current_question is not None and not answer_seen_for_current:
            o_match = option_pattern.match(stripped)
            opt_letter = None
            opt_text = None

            if o_match:
                opt_letter = o_match.group(1)
                opt_text = o_match.group(2).strip()
            else:
                # "A)" gibi tek başına harf mi? Sonraki satır seçenek metni olabilir
                loose_match = option_pattern_loose.match(stripped)
                if loose_match:
                    opt_letter = loose_match.group(1)
                    # Sonraki satırı seçenek metni olarak al
                    if i + 1 < len(cleaned_lines):
                        next_line = cleaned_lines[i + 1].strip()
                        if next_line and not question_start_pattern.match(next_line) \
                                and not option_pattern.match(next_line) \
                                and not correct_answer_pattern.search(next_line):
                            opt_text = next_line
                            i += 1  # Sonraki satırı da tükettik

            if opt_letter and opt_text:
                opt_text = re.sub(r'Most\s+Voted', '', opt_text, flags=re.IGNORECASE).strip()

                # Çok satırlı seçenek metinlerini birleştir
                while i + 1 < len(cleaned_lines):
                    next_line = cleaned_lines[i + 1].strip()
                    if not next_line:
                        break
                    if question_start_pattern.match(next_line):
                        break
                    if option_pattern.match(next_line):
                        break
                    if option_pattern_loose.match(next_line):
                        break
                    if correct_answer_pattern.search(next_line):
                        break
                    if next_line.startswith(('Explanation:', 'Correct Answer', 'Answer', 'Ans')):
                        break
                    # Açıklama paragrafı gibi görünen satırları durdur
                    if re.match(r'^[A-Z][a-z]', next_line) and len(next_line) > 80:
                        break
                    if re.match(r'^(The|This|These|Those|It|In|For|A |An )', next_line) and len(next_line) > 60:
                        break
                    opt_text += " " + next_line
                    i += 1

                current_options.append(f"{opt_letter}. {opt_text}")
                i += 1
                continue

            # Soru metninin devamı (henüz seçenek başlamadıysa)
            if len(current_options) == 0:
                if not stripped.startswith(('Explanation:', 'Correct Answer', 'Answer', 'Ans')):
                    if not correct_answer_pattern.search(stripped):
                        current_question += " " + stripped

        i += 1

    save_current()

    # ============================================================
    # STAGE 2: Build final questions and MERGE duplicates by number.
    # ============================================================
    questions_by_num = {}
    skipped_questions = []

    for rq in raw_questions:
        # 2'den az seçeneği olan soruları atla ve raporla
        if len(rq['options']) < 2:
            skipped_questions.append(rq)
            continue

        q_num = rq['num']
        if not q_num:
            skipped_questions.append(rq)
            continue

        if q_num not in questions_by_num:
            questions_by_num[q_num] = rq
        else:
            existing_has_answer = bool(questions_by_num[q_num]['correct'])
            new_has_answer = bool(rq['correct'])
            if new_has_answer and not existing_has_answer:
                questions_by_num[q_num] = rq

    # ============================================================
    # DEBUG ÇIKTILARI - Terminalde görünecek
    # ============================================================
    if debug:
        print("\n" + "=" * 70)
        print("=== PDF PARSING DEBUG RAPORU ===")
        print("=" * 70)
        print(f"Toplam ham (raw) soru sayısı       : {len(raw_questions)}")
        print(f"Benzersiz (indekslenen) soru sayısı: {len(questions_by_num)}")
        print(f"Atlanan soru sayısı                : {len(skipped_questions)}")
        print(f"Fark (elenen duplicate)            : {len(raw_questions) - len(skipped_questions) - len(questions_by_num)}")
        print("=" * 70)

        # İlk 30 atlanan soruyu göster
        print("\n--- ATLANAN SORULARIN İLK 30'U ---\n")
        for s in skipped_questions[:30]:
            print(f"Soru No: {s['num'] if s['num'] else '(yok)'} | "
                  f"Seçenek Sayısı: {len(s['options'])}")
            print(f"  Metin: {s['text'][:150]}")
            for opt in s['options']:
                print(f"    -> {opt[:100]}")
            if not s['options']:
                print("    (SEÇENEK YOK!)")
            print("-" * 50)

    # Final listeyi oluştur
    final_questions = []
    for q_num, rq in questions_by_num.items():
        correct_full = ""
        if rq['correct']:
            correct_clean = rq['correct'].replace(" ", "").replace(",", "").upper()
            if correct_clean:
                first_letter = correct_clean[0]
                for opt in rq['options']:
                    if opt.strip().startswith(first_letter + "."):
                        correct_full = opt
                        break
                if not correct_full:
                    correct_full = correct_clean

        if rq['num']:
            soru_text = f"{rq['num']}) {rq['text']}"
        else:
            soru_text = rq['text']

        final_questions.append({
            'soru': soru_text,
            'siklar': rq['options'],
            'dogru_cevap': correct_full
        })

    # Soru numarasına göre sırala
    def sort_key(q):
        m = re.match(r'^(\d+)\)', q['soru'])
        return int(m.group(1)) if m else 999999

    final_questions.sort(key=sort_key)

    return final_questions


# ------------------------------------------------------------------
#  Database and Model Functions
# ------------------------------------------------------------------
@st.cache_resource
def get_embedding_model():
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Embedding model loaded.")
    return model

@st.cache_resource
def get_vector_db():
    print("Initializing vector database...")
    client = chromadb.EphemeralClient()
    print("Vector database initialized.")
    return client

def setup_database(client, model, questions_list):
    try:
        collection = client.get_or_create_collection(name="aws_questions")
    except Exception as e:
        st.error(f"Could not create vector DB collection: {e}")
        return None

    if collection.count() != len(questions_list):
        print(f"Database contains {collection.count()} / {len(questions_list)} questions. Re-indexing...")
        if collection.count() > 0:
            client.delete_collection(name="aws_questions")
            collection = client.get_or_create_collection(name="aws_questions")

        documents_to_embed = []
        metadatas_for_db = []
        ids_for_db = []

        for i, q in enumerate(questions_list):
            content = f"Question: {q['soru']}"
            documents_to_embed.append(content)
            metadatas_for_db.append({"original_index": i})
            ids_for_db.append(f"q_{i}")

        batch_size = 500
        for start in range(0, len(documents_to_embed), batch_size):
            end = start + batch_size
            embeddings = model.encode(documents_to_embed[start:end])
            collection.add(
                embeddings=embeddings.tolist(),
                documents=documents_to_embed[start:end],
                metadatas=metadatas_for_db[start:end],
                ids=ids_for_db[start:end]
            )
        print("Indexing complete.")
    else:
        print("Database is already up to date. Skipping indexing.")

    return collection

@st.cache_data
def load_and_parse_questions(pdf_path):
    print("Parsing PDF...")
    if not os.path.exists(pdf_path):
        st.error(f"Error: PDF not found at '{pdf_path}'.")
        return None
    try:
        with open(pdf_path, "rb") as f:
            return parse_aws_questions(f, debug=True)
    except Exception as e:
        st.error(f"PDF read error: {e}")
        return None

# ------------------------------------------------------------------
#  Streamlit Interface
# ------------------------------------------------------------------

st.title("AWS CP2 Quiz Bot 🧠☁️")

# --- 1. Loading and Setup ---
model = get_embedding_model()
client = get_vector_db()
questions_list = load_and_parse_questions("data/aws_sorulari.pdf")

if not questions_list:
    st.error("The application was stopped because no questions could be read from the PDF.")
    st.stop()

collection = setup_database(client, model, questions_list)
if not collection:
    st.error("The application was stopped because the vector database could not be set up.")
    st.stop()

# --- Session State ---
if 'quiz_started' not in st.session_state:
    st.session_state.quiz_started = False
    st.session_state.questions_to_ask = []
    st.session_state.num_to_ask = 0
    st.session_state.current_question_index = 0
    st.session_state.score = 0
    st.session_state.user_answers = {}

# Stage 1: Select Topic OR Random
if not st.session_state.quiz_started:

    st.info(f"{collection.count()} AWS questions indexed.")
    st.caption(f"PDF'ten ayrıştırılan toplam soru: {len(questions_list)}")

    user_topic = st.text_input(
        "Which topic do you want to be quizzed on? (Leave blank for random)",
        placeholder="e.g., S3 and storage"
    )

    num_to_ask_input = st.number_input(
        "How many questions?",
        min_value=1,
        max_value=50,
        value=3
    )
    num_to_ask = int(num_to_ask_input)

    if st.button("Start Quiz"):

        retrieved_questions = []

        if user_topic.strip():
            with st.spinner(f"Finding the {num_to_ask} best questions about '{user_topic}'..."):
                query_embedding = model.encode([user_topic])[0].tolist()
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=num_to_ask
                )
                for metadata in results['metadatas'][0]:
                    idx = metadata['original_index']
                    retrieved_questions.append(questions_list[idx])

        else:
            with st.spinner(f"Selecting {num_to_ask} random questions..."):
                num_available = len(questions_list)
                actual_num_to_get = min(num_to_ask, num_available)
                if actual_num_to_get > 0:
                    retrieved_questions = random.sample(questions_list, actual_num_to_get)

        if not retrieved_questions:
            st.warning("No questions found. Please check your topic or if the PDF was parsed correctly.")
        else:
            st.session_state.questions_to_ask = retrieved_questions
            st.session_state.num_to_ask = len(retrieved_questions)
            st.session_state.quiz_started = True
            st.session_state.current_question_index = 0
            st.session_state.score = 0
            st.session_state.user_answers = {}
            st.rerun()

# Stage 2: Show Quiz
elif st.session_state.quiz_started and st.session_state.current_question_index < st.session_state.num_to_ask:

    idx = st.session_state.current_question_index
    q = st.session_state.questions_to_ask[idx]

    st.subheader(f"Question {idx + 1} / {st.session_state.num_to_ask}")
    st.write(q.get('soru', 'Question text not found'))

    with st.form(key=f"form_q_{idx}"):
        user_answer = st.radio(
            "Select your answer:",
            q.get('siklar', []),
            key=f"radio_q_{idx}",
            index=None
        )
        submit_button = st.form_submit_button("Submit Answer")

    if submit_button:
        if user_answer is None:
            st.warning("Please select an answer.")
        else:
            st.session_state.user_answers[idx] = user_answer

            correct_answer_text = q.get('dogru_cevap', '').strip()
            user_answer_prefix = user_answer.strip()[0] if user_answer.strip() else ""

            correct_letter = ""
            if correct_answer_text:
                if len(correct_answer_text) > 5 and correct_answer_text[1:2] == '.':
                    correct_letter = correct_answer_text[0]
                else:
                    correct_letter = correct_answer_text.replace(" ", "").replace(",", "").upper()

            if correct_letter and user_answer_prefix in correct_letter:
                st.success("Correct! 🎉")
                st.session_state.score += 1
            else:
                if correct_answer_text:
                    st.error(f"❌ Incorrect. The correct answer was: **{correct_answer_text}**")
                else:
                    st.error("❌ Incorrect. (The correct answer could not be extracted from the PDF for this question.)")

            st.caption(f"Your answer: {user_answer}")

            st.session_state.current_question_index += 1

            if st.session_state.current_question_index < st.session_state.num_to_ask:
                st.button("Next Question")
            else:
                st.button("View Results")

# Stage 3: Results Screen
elif st.session_state.quiz_started and st.session_state.current_question_index >= st.session_state.num_to_ask:
    st.balloons()
    st.header("Quiz Finished!")
    st.write(f"You answered {st.session_state.score} out of {st.session_state.num_to_ask} questions correctly.")

    if st.button("Start Over"):
        st.session_state.quiz_started = False
        st.session_state.questions_to_ask = []
        st.session_state.num_to_ask = 0
        st.session_state.current_question_index = 0
        st.session_state.score = 0
        st.session_state.user_answers = {}
        st.rerun()