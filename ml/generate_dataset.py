"""
generate_dataset.py
-------------------
Generates a synthetic but realistic document classification dataset with at
least 1,500 rows across seven document categories.  Each example is built by
randomly assembling sentences from category-specific phrase banks so no two
examples are identical.

Output: ml/data/document_dataset.csv
Columns: text, label
"""

import random
import csv
import os
from pathlib import Path

SEED = 42
random.seed(SEED)

LABELS = [
    "syllabus",
    "assignment_sheet",
    "event_flyer",
    "receipt",
    "scholarship_notice",
    "job_posting",
    "general_notes",
]

# ---------------------------------------------------------------------------
# Phrase banks — each list contains realistic sentence fragments for the class.
# Examples are assembled by sampling 4–10 sentences so every row is unique.
# ---------------------------------------------------------------------------

PHRASE_BANKS: dict[str, list[str]] = {
    "syllabus": [
        "Welcome to Introduction to Computer Science, Section 003.",
        "This course meets Monday, Wednesday, and Friday from 10:00 AM to 10:50 AM in Room 214.",
        "Office hours are held every Tuesday from 2:00 PM to 4:00 PM in Building C, Room 118.",
        "The instructor can be reached at professor.harris@university.edu.",
        "Attendance is mandatory; more than three unexcused absences will lower your grade by one letter.",
        "Late assignments will be penalized 10% per day up to a maximum of three days.",
        "Grading: Homework 30%, Midterm 25%, Final Exam 35%, Participation 10%.",
        "The midterm exam is scheduled for October 15 during regular class time.",
        "The final exam will be held during the officially scheduled finals period in December.",
        "Academic integrity violations will be reported to the Dean of Students.",
        "Required textbook: 'Introduction to Algorithms' by Cormen et al., 4th edition.",
        "All assignments must be submitted through the course portal by 11:59 PM on the due date.",
        "Students needing accommodations should contact the Disability Resource Center.",
        "Course topics include data structures, algorithm analysis, sorting, and graph theory.",
        "Participation credit requires active engagement during in-class discussions.",
        "There will be six homework assignments, each worth equal weight.",
        "Quiz dates will be announced one week in advance on the course website.",
        "The syllabus is subject to change; updates will be posted to the LMS.",
        "Email response time is typically 24–48 hours on weekdays.",
        "Group projects are permitted with prior written approval from the instructor.",
        "No make-up exams will be given except in cases of documented medical emergency.",
        "Chapter readings should be completed before each Tuesday lecture.",
        "Laptop use is permitted for note-taking but not for unrelated browsing.",
        "All written work must be in 12-point Times New Roman with one-inch margins.",
        "Students are encouraged to form study groups and use the university tutoring center.",
        "The drop deadline for this course is September 20.",
        "Course prerequisite: MATH 1101 or placement into MATH 1501.",
        "Extra credit opportunities may be announced at the instructor's discretion.",
        "Grading scale: A 90–100, B 80–89, C 70–79, D 60–69, F below 60.",
        "All exams are closed-book unless otherwise stated.",
        "Students must bring a government-issued photo ID to every exam.",
        "Office location: College of Computing, Room 418.",
        "The course uses Python 3.10 and Jupyter Notebook for programming assignments.",
        "Weekly lab sessions meet on Thursdays and are worth 5% of the final grade.",
        "Collaboration on individual assignments is not permitted.",
        "This course satisfies the university's quantitative reasoning requirement.",
    ],
    "assignment_sheet": [
        "Assignment 3: Sentiment Analysis Classifier — Due November 8 at 11:59 PM.",
        "Submit your work as a single ZIP file named Lastname_Firstname_HW3.zip.",
        "Your submission must include a PDF report (minimum 4 pages) and all source code.",
        "The report should describe your methodology, experimental results, and conclusions.",
        "Code must be written in Python 3.10; no other languages are accepted for this assignment.",
        "You must use at least two classification algorithms and compare their performance.",
        "Report your accuracy, precision, recall, and F1 score for each model.",
        "Include a confusion matrix visualization in your report.",
        "Your dataset should contain at least 500 labeled examples.",
        "A rubric is attached at the end of this document; read it carefully before starting.",
        "Points will be deducted for missing sections, late submission, or poor code quality.",
        "Grading breakdown: Written report 40 pts, Code quality 30 pts, Results 30 pts.",
        "Starter code and example data are available on the course GitHub repository.",
        "Do not share your solution with other students; this is an individual assignment.",
        "Extra credit: Implement a third model and include a statistical significance test.",
        "Page limit: Your report may not exceed 10 pages excluding references and appendices.",
        "Use IEEE citation format for all references.",
        "Your code must run without errors in a clean virtual environment.",
        "Include a requirements.txt file listing all Python dependencies.",
        "The TA will hold a special office hour on November 6 to answer questions.",
        "Deliverables: report.pdf, source_code/, requirements.txt, and README.md.",
        "Late submissions: -10 points per day; no submissions accepted after November 11.",
        "Peer review is not required for this assignment but is encouraged.",
        "Your model must be trained and tested on separate data splits.",
        "Do not use any pre-trained neural network models for this assignment.",
        "Submit via the course portal under the Assignment 3 submission link.",
        "All figures must include axis labels, titles, and captions.",
        "Word count for the report introduction: minimum 250 words.",
        "You may use scikit-learn, pandas, and matplotlib for this assignment.",
        "Describe any ethical considerations of deploying your model in a real application.",
        "Bring a printed copy of your report to class on November 9 for discussion.",
        "Attach your signed academic integrity statement to the submission.",
        "File naming convention must be followed exactly or points will be deducted.",
        "The rubric awards partial credit for incomplete but reasoned attempts.",
        "Evaluation environment: Python 3.10, scikit-learn 1.3, pandas 2.0.",
    ],
    "event_flyer": [
        "Join us for the Spring Networking Social hosted by the Women in Tech Society!",
        "Date: Friday, April 18 | Time: 6:00 PM – 8:30 PM",
        "Location: Student Union Ballroom, Room 201",
        "RSVP by April 14 at bit.ly/wits-spring-rsvp — space is limited!",
        "Enjoy light refreshments, speed networking, and a panel of guest speakers.",
        "Guest speaker: Maria Chen, Senior Software Engineer at Apple.",
        "Open to all undergraduate and graduate students — bring your resume!",
        "Dress code: Business casual.",
        "Hosted by the Association for Computing Machinery Student Chapter.",
        "Free event — no registration fee required.",
        "Workshop: How to Ace Technical Interviews — Saturday, March 22, 2:00 PM",
        "Co-sponsored by the Career Development Center and the CS Department.",
        "Questions? Email events@studentorg.edu or visit our Instagram @cs_events.",
        "Annual Hackathon: 24 hours to build something amazing.",
        "Prizes: $500 first place, $300 second place, $150 third place.",
        "Registration closes February 28 — register at hackathon.university.edu.",
        "Teams of 2–4 students; individual sign-ups will be matched with teammates.",
        "Mentors from local tech companies will be available throughout the event.",
        "Free pizza and snacks will be provided throughout the hackathon.",
        "Join the AI Study Group — Wednesdays at 5 PM in the Library Collaboration Room.",
        "Data Science Career Panel — March 5, 4:00 PM – 6:00 PM, Engineering Hall 110.",
        "Featuring professionals from Google, Meta, and Bloomberg.",
        "Attend the Annual Tech Fair and connect with 30+ companies hiring interns.",
        "Bring 20 copies of your resume to the career fair.",
        "The event is wheelchair accessible; contact us if you need accommodations.",
        "Refreshments sponsored by the Student Government Association.",
        "Scan the QR code on this flyer to add the event to your calendar.",
        "Photography and video may be taken at this event for promotional purposes.",
        "Volunteer opportunities available — email volunteer@techfair.edu.",
        "Check out our website for full event schedule: studenttech.university.edu.",
        "This event is proudly supported by the College of Engineering.",
        "After-party at the on-campus coffee shop immediately following the panel.",
        "Certificate of participation will be provided upon request.",
        "Limited parking available in Lot B; carpool or take the campus shuttle.",
        "Follow us on LinkedIn: linkedin.com/company/cs-student-org.",
    ],
    "receipt": [
        "Campus Bookstore — Thank you for your purchase!",
        "Transaction ID: TXN-20241105-88371",
        "Date: November 5, 2024 | Time: 2:34 PM",
        "Item: Introduction to Algorithms (Textbook) — $89.99",
        "Item: Composition Notebook (2-pack) — $6.49",
        "Item: USB-C Hub — $24.99",
        "Item: Blue Pens (12-pack) — $3.99",
        "Subtotal: $125.46",
        "Tax (8.5%): $10.66",
        "Total: $136.12",
        "Payment Method: Visa ending in 4821",
        "Cashier: Employee #047",
        "Returns accepted within 30 days with receipt.",
        "Campus Coffee Co. | Order #1024 | April 3, 2025",
        "Item: Large Oat Milk Latte — $6.50",
        "Item: Blueberry Muffin — $3.75",
        "Total: $10.25 | Paid: Apple Pay",
        "Thank you! Your feedback helps us improve — review us on Yelp.",
        "University Parking Services — Citation Payment Receipt",
        "Citation #: CIT-2024-00451 | Amount Paid: $35.00",
        "Payment Date: September 12, 2024 | Reference: PAY-981234",
        "Online Store — Order Confirmation",
        "Order #: ORD-556778 | Shipped to: 123 Dorm Hall, Atlanta, GA 30303",
        "Item: Laptop Stand — $42.00 | Shipping: $0.00 (Prime)",
        "Estimated Delivery: November 10, 2024",
        "Grocery Total: $47.83 | Store: Campus Market",
        "Items: Ramen x4, Instant Oatmeal x2, Energy Drink x3, Bananas, Bread",
        "Payment: EBT/SNAP card | Balance After Purchase: $52.17",
        "Dining Hall — Meal Plan Transaction",
        "Student ID: 90123456 | Meal Swipe Used | Balance: 14 swipes remaining",
        "Date: October 22, 2024 | Location: North Dining Hall",
        "Subscription Renewal: Spotify Premium — $5.99/month (Student Plan)",
        "Billing Date: November 1, 2024 | Card: Mastercard ending in 7744",
        "Confirmation Number: SPOT-2024-881734",
        "Laundry Services — $3.50 deducted from student card | Machine #12 | Wash",
        "Print Services — 48 pages x $0.10 = $4.80 | Student ID: 90123456",
    ],
    "scholarship_notice": [
        "The Georgia STEM Excellence Scholarship is now accepting applications.",
        "Award Amount: Up to $5,000 per academic year.",
        "Eligibility: Full-time undergraduate students with a minimum GPA of 3.5.",
        "Applicants must be enrolled in a STEM degree program.",
        "Application Deadline: March 15, 2025 at 11:59 PM.",
        "Required materials: personal statement, official transcript, and two letters of recommendation.",
        "Submit all application materials electronically to scholarships@gsu.edu.",
        "Incomplete applications will not be considered.",
        "Award recipients will be notified by April 30, 2025.",
        "This scholarship is renewable for up to four years with continued eligibility.",
        "The scholarship is funded by the Georgia Tech Alumni Foundation.",
        "Questions? Contact the Financial Aid Office at financialaid@gsu.edu or call (404) 555-0192.",
        "Priority consideration will be given to first-generation college students.",
        "Applicants must demonstrate financial need as determined by the FAFSA.",
        "The personal statement should be between 500 and 750 words.",
        "Letters of recommendation must be submitted directly by the recommender.",
        "Transcripts must be official and sent directly from the registrar.",
        "Recipients must maintain a 3.3 GPA each semester to retain the award.",
        "The award may be used for tuition, fees, books, and housing.",
        "No separate FAFSA is required; financial aid data is pulled automatically.",
        "Apply online at scholarships.university.edu/stem-excellence.",
        "This award is not stackable with other university merit scholarships over $3,000.",
        "International students are not eligible for this particular scholarship.",
        "Part-time students enrolled in fewer than 12 credit hours are not eligible.",
        "The scholarship committee will conduct interviews with finalists in April.",
        "All interviewed candidates will receive a decision letter by May 1.",
        "Contact your academic advisor to confirm your eligibility before applying.",
        "Attend the Financial Aid Scholarship Workshop on February 20 for application tips.",
        "Early submissions are encouraged; applications are reviewed on a rolling basis.",
        "The scholarship committee values community involvement and leadership experience.",
        "Attach a current resume highlighting academic and extracurricular achievements.",
        "This notice is distributed by the Office of Student Financial Services.",
        "Award disbursement occurs at the start of each semester upon enrollment verification.",
        "Students may hold this scholarship simultaneously with need-based grants.",
        "For accessibility accommodations during the interview process, contact our office.",
    ],
    "job_posting": [
        "Software Engineering Intern — Apple, Cupertino, CA (Summer 2025)",
        "We are looking for motivated students to join the Core ML team for a 12-week internship.",
        "Responsibilities: Design and implement machine learning features for iOS frameworks.",
        "Qualifications: Currently pursuing a BS or MS in Computer Science, ML, or related field.",
        "Required skills: Python, Swift, Core ML, PyTorch or TensorFlow, Git.",
        "Preferred skills: Experience with on-device inference, model compression, or CoreData.",
        "GPA requirement: 3.0 or above preferred.",
        "Applications due: January 31, 2025 — apply at jobs.apple.com/internship.",
        "This is a paid internship; compensation is competitive and commensurate with experience.",
        "Interns will work directly with full-time engineers on shipped products.",
        "Full-Time Role: Data Scientist — FinTech Analytics, Remote",
        "Join a fast-growing team building AI-powered financial insights products.",
        "Minimum 2 years of experience with Python, pandas, scikit-learn, and SQL.",
        "Experience with A/B testing, dashboards, and stakeholder communication required.",
        "Responsibilities include designing experiments, analyzing results, and presenting findings.",
        "Salary Range: $95,000 – $130,000 depending on experience.",
        "Benefits: Health, dental, vision, 401k with 4% match, flexible PTO.",
        "Apply by submitting your resume and a 1-page cover letter to careers@fintechco.com.",
        "Machine Learning Engineer — Healthcare AI Startup, Atlanta, GA",
        "Build and deploy models that assist clinicians with diagnostic support tools.",
        "Required: 3+ years of Python, PyTorch, model deployment, and REST APIs.",
        "Familiarity with HIPAA compliance and privacy-preserving ML is a strong plus.",
        "Work with cross-functional teams including doctors, engineers, and product designers.",
        "Equity options available for early hires.",
        "Product Manager, AI Features — Consumer Tech Company",
        "Define product strategy for AI-powered features used by millions of users.",
        "Must have: 5 years of product management experience, strong analytical skills.",
        "Bachelor's degree required; MBA or technical degree preferred.",
        "Skills: user research, roadmapping, cross-functional collaboration, data analysis.",
        "Research Engineer Intern — NLP Team",
        "Conduct experiments on large language model fine-tuning and evaluation.",
        "Strong background in NLP, transformers, and evaluation metrics required.",
        "Publish findings internally and contribute to external research collaborations.",
        "PhD students preferred; exceptional MS students will be considered.",
        "To apply: submit resume, cover letter, and GitHub portfolio to nlp-hiring@techlab.com.",
    ],
    "general_notes": [
        "Meeting notes — October 3",
        "Discussed sprint goals for Q4; need to finalize API design by end of week.",
        "TODO: follow up with Marcus about the database migration timeline.",
        "Key insight: users are dropping off at the payment screen — investigate UX.",
        "Remember to review the pull request for the authentication module.",
        "Class notes: Machine Learning Lecture 9",
        "Overfitting occurs when a model performs well on training data but poorly on test data.",
        "Regularization techniques include L1 (Lasso) and L2 (Ridge) penalties.",
        "Cross-validation helps estimate model generalization without a fixed test set.",
        "Study group reminder: meet in the library at 6 PM Thursday.",
        "Personal brainstorm — app ideas",
        "Idea 1: a habit tracker that uses ML to predict when you're likely to skip a day.",
        "Idea 2: a tool that summarizes research papers into one-paragraph summaries.",
        "Idea 3: an app that categorizes receipts and tracks spending automatically.",
        "Need to research: Core ML model size limits for iPhone storage.",
        "Reading notes: Chapter 5 — Neural Networks",
        "A perceptron is the basic unit; it computes a weighted sum plus a bias term.",
        "Activation functions like ReLU and sigmoid introduce non-linearity.",
        "Backpropagation uses the chain rule to compute gradients layer by layer.",
        "The learning rate controls how large each gradient step is.",
        "Reminder: submit internship application by Friday.",
        "Buy groceries: eggs, oat milk, bread, apples, peanut butter.",
        "Call mom this weekend — haven't talked in two weeks.",
        "Project meeting moved to Wednesday at 3 PM in the engineering lounge.",
        "Interesting fact: the human brain has approximately 86 billion neurons.",
        "Questions to ask during office hours: bias-variance tradeoff, PCA intuition.",
        "Draft outline for capstone proposal: intro, problem statement, methods, timeline.",
        "To-do before Monday: finish problem set 4, read chapter 7, email professor.",
        "Random thought: could combine GPS data with study behavior to predict final grades.",
        "Journal: today was productive — finished the feature extractor and started training.",
        "Notes from advising meeting: register for spring semester by November 1.",
        "Career fair tips: prepare 30-second elevator pitch, bring 20 resumes.",
        "Useful Python libraries to explore: polars, duckdb, plotly, streamlit.",
        "Workout log: ran 3 miles, 50 push-ups, 30 minutes yoga.",
        "The most important thing I learned this week: debug by simplifying, not by adding.",
    ],
}

# ---------------------------------------------------------------------------
# Sentence templates that inject dynamic values (dates, emails, amounts, etc.)
# to increase variety within each class.
# ---------------------------------------------------------------------------

DATES = [
    "January 10", "February 3", "March 15", "April 22", "May 7",
    "September 5", "October 14", "November 30", "December 1", "August 19",
    "January 31", "February 28", "March 8", "April 1", "June 15",
]

EMAILS = [
    "professor.jones@university.edu", "scholarships@gsu.edu",
    "careers@company.com", "ta.support@cs.edu", "events@studentorg.edu",
    "admissions@college.edu", "financialaid@university.edu",
    "internships@techcorp.com", "hiring@startup.io",
]

AMOUNTS = ["$12.50", "$89.99", "$250.00", "$4,500", "$5,000", "$135.00",
           "$22.75", "$3.99", "$1,200", "$500", "$47.83", "$6.50"]

INSTRUCTORS = [
    "Dr. Smith", "Professor Nguyen", "Dr. Patel", "Professor Williams",
    "Dr. Chen", "Professor Harris", "Dr. Jackson",
]

COMPANIES = [
    "Apple", "Google", "Meta", "Microsoft", "Amazon", "IBM",
    "Salesforce", "Adobe", "NVIDIA", "Stripe",
]

ROLES = [
    "Software Engineering Intern", "Machine Learning Engineer",
    "Data Science Intern", "iOS Developer", "Research Engineer",
    "Product Manager", "Backend Engineer", "Computer Vision Engineer",
]


def _pick(lst: list, k: int = 1) -> list:
    """Return k random unique elements from lst."""
    k = min(k, len(lst))
    return random.sample(lst, k)


def _dynamic_sentences(label: str) -> list[str]:
    """Generate a small list of dynamic sentences specific to each label."""
    sentences = []
    date = random.choice(DATES)
    email = random.choice(EMAILS)
    amount = random.choice(AMOUNTS)
    instructor = random.choice(INSTRUCTORS)
    company = random.choice(COMPANIES)
    role = random.choice(ROLES)

    if label == "syllabus":
        sentences += [
            f"The course instructor is {instructor}.",
            f"Contact the instructor at {email} for questions.",
            f"Midterm exam date: {date}.",
            f"Final project due: {date} at 11:59 PM.",
        ]
    elif label == "assignment_sheet":
        sentences += [
            f"This assignment is due on {date} at midnight.",
            f"Submit your work to {email}.",
            f"Late penalty: deduct 10 points per day after {date}.",
            f"Attach your grading rubric; total points: 100.",
        ]
    elif label == "event_flyer":
        sentences += [
            f"Event date: {date} — mark your calendar!",
            f"RSVP to {email} by {date}.",
            f"Featured speaker from {company}.",
            f"Refreshments and networking begin at 5:30 PM.",
        ]
    elif label == "receipt":
        sentences += [
            f"Transaction completed on {date}.",
            f"Total charged: {amount}.",
            f"For questions about this transaction, email {email}.",
            f"Item total before tax: {amount}.",
        ]
    elif label == "scholarship_notice":
        sentences += [
            f"Application deadline: {date}.",
            f"Send all materials to {email}.",
            f"Award value: {amount} per academic year.",
            f"Eligibility decisions will be communicated by {date}.",
        ]
    elif label == "job_posting":
        sentences += [
            f"Position: {role} at {company}.",
            f"Application deadline: {date}.",
            f"Submit resume and cover letter to {email}.",
            f"Compensation: {amount} (internship stipend or monthly salary).",
        ]
    elif label == "general_notes":
        sentences += [
            f"Reminder set for {date}.",
            f"Contact {email} about the follow-up.",
            f"Estimated cost: {amount}.",
            f"Meeting with {instructor} scheduled for {date}.",
        ]
    return sentences


def generate_example(label: str) -> str:
    """Assemble one document example by sampling the phrase bank + dynamic sentences."""
    bank = PHRASE_BANKS[label]
    n_static = random.randint(4, 9)
    n_dynamic = random.randint(1, 3)

    static_lines = _pick(bank, n_static)
    dynamic_lines = _pick(_dynamic_sentences(label), n_dynamic)

    all_lines = static_lines + dynamic_lines
    random.shuffle(all_lines)

    separator = random.choice([" ", "\n", " — "])
    text = separator.join(all_lines)

    # light noise: occasional ALL CAPS header or bullet prefix
    prefix_choices = ["", "", "", ""]
    prefix = random.choice(prefix_choices)
    if prefix:
        text = f"{prefix}\n{text}"

    return text.strip()


def build_dataset(n_per_class: int = 220) -> list[dict]:
    """
    Build the full dataset.  n_per_class * 7 classes = total rows.
    Default gives 1,540 rows, exceeding the 1,500-row requirement.
    """
    rows = []
    for label in LABELS:
        for _ in range(n_per_class):
            text = generate_example(label)
            rows.append({"text": text, "label": label})

    random.shuffle(rows)
    return rows


def main() -> None:
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "document_dataset.csv"

    print("Generating synthetic document dataset …")
    rows = build_dataset(n_per_class=220)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)

    label_counts: dict[str, int] = {}
    for row in rows:
        label_counts[row["label"]] = label_counts.get(row["label"], 0) + 1

    print(f"\nDataset saved to: {output_path}")
    print(f"Total rows: {len(rows)}\n")
    print("Rows per class:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label:<22} {count}")


if __name__ == "__main__":
    main()
