INTRODUCTION
There are 700,000 rheumatoid arthritis (RA) sufferers in India, and this number rises by 30,000
per year. RA-related joint deterioration causes discomfort, decreased joint function, joint
degeneration, and joint deformity. Early intervention enhances the prognosis, however it is
crucial to accurately gauge the rate of RA progression and recommend the best course of action.Xrays of the hand or foot are used in the diagnosis of RA. The modified Total Sharp (mTS) score
is utilized to assess joint space narrowing (JSN) and erosion in the 32 hand joints and 12 foot
joints. The 4 grades of the JSN score and the 5 grades of the abrasion score are personally
assigned to each joint by orthopedic doctors.
1.1 Project overview
However, X-ray images must be taken multiple times year to enable accurate assessment
numerous because there are different evaluation points for the MTS score and it might be
challenging to communicate the results. Because it determined by orthopedicians, score is
likewise subjective. It is consequently necessary to implement an automated MTS score
calculation method supported by X-ray image analysis. Technology for automatically
recognizing finger joints is necessary for the fully automated score computation method.
Reference [2] suggests a finger joint detection system based on deep learning. It can only
applied to growing children's finger joints; it cannot be use on RA patients directly. Becausethe
compressed finger joints of patients with severe RA are too tiny for joint examination, TheMTS
score examines the erosion score and JSN score for each finger joint. For a patient with mild
RA, Reference [3] automatically determines the JSN score.
1.2 Aim
The effectiveness of a fully automated joint detection and mTS score calculation method is
assessed in this study. Additionally, we see a specific stage of performance increase by
artificially rotating and gamma correcting the training image. For clinical application, we
additionally evaluate total MtS and information about estimated scores.


1.3 Objective
• The study objective of this is to evaluate a fully automated finger jointidentification and
MTs score estimate method.
• In addition, by artificially rotating and gamma-correcting the training image, we observe an
increase in performance at a particular level.
• In addition, we assesstotal mTS and information on anticipated scoresfor clinical application