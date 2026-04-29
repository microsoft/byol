run the diff checker between current repo and REF repo. 
Ref repo is this: /home/szamir/ssdprivate/projects/low-resource-llm-toolkit/translators_refactor/byol-review 
--you need to make sure to see which files are changed 
-- Which files are deleted from the Ref repo 

Make sure run proper diff check at the file content level as well. 

Give me a summary and table listing the files which are changed.
Remeber Ref is the source of truth and the table should should what is in ref and what is in curr in the modified files.


Ok copy over these modifed files from REF to current codebase and overwirte here. 

I am going to release this code base along with my research paper can you make a critical analysis of this code base and tell me if it is ready to be released as a research artifact so the other users can use it to easily build LLMS for their own languages. 
Remember this is a research repo I do not want it to be over complex I do not want to be over engineered just tell me if which parts are good which parts are absolutely disaster.
Add all your critical analysis in analysis.md file. 


Look, I am going to release this codebase on Github for my research paper BYOL.  It is a research toolkit so the other users can use it to easily build LLMs for their own languages. As part of the paper, we trained and evaluated models for two langauges, Chichewa (nya) and Maori (mri). We basically, trained models both continual trained models and instruction trained models and also the created merged models. The CPT and Merged models (M) are then evaluated on different benchmarks and results were presented in the paper. 
The trained models are provided at this link: /home/szamir/shared/ai4g-auh/byol_weights. These trained weights I will basically upload here: https://huggingface.co/ai-for-good-lab

Also, as part of our work/contributions, we will also release human trasnalted version of Global MMLU-Lite for  lanagues: Inuktitut, Chichewa, Maori. We will also upload them here: https://huggingface.co/ai-for-good-lab

Now my question to you is. Considering the context above, what do you think about the top level readme. 


Do you think adding something like this is a good strtegy:

#### News
- **April 4, 2022:** Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the web demo: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/swzamir/Restormer)
- **March 30, 2022:** Added Colab Demo. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1C2818h7KnjNv4R1sabe14_AYL7lWhmu6?usp=sharing)
- **March 29, 2022:** Restormer is selected for an ORAL presentation at CVPR 2022 :dizzy:
- **March 10, 2022:** Training codes are released :fire:
- **March 3, 2022:** Paper accepted at CVPR 2022 :tada: 
- **Nov 21, 2021:** Testing codes and pre-trained models are released!

So keeping the current readme and adding news section will address your concerns you highlighted before?

can you create a README_DRAFT.md for me to analyze make sure whatever hugging face links you metnion in there, you list them and guide me what to upload and where and how to structure that, so I can then follow those steps and upload the the relevant artifacts on the hugging face.

I forgot to mention, here is the path Global-MMLU-Lite for langauges. 
~/shared/ai4g-auh/global-mmlu-lite-translated/
Chichewa
GlobalMMLU-Lite_Dev_Chichewa.json
GlobalMMLU-Lite_Test_Chichewa.json
 
Inuktitut
GlobalMMLU-Lite_Dev_Inuktitut.json
GlobalMMLU-Lite_Test_Inuktitut.json
 
Maori
GlobalMMLU-Lite_Dev_Maori.json
GlobalMMLU-Lite_Test_Maori.json


OK now I want to if you see my get history there are two branches you have main branch and initial feature branch the initial feature branch is is the one which is reviewed by the engineering team and asked me to make some modifications so I made those modifications I would like to push these changes to the code base now so give me a step by step guide to push it.

Basiclaly At the end I want to merge those branches into one before pushing. So that my AzureDevops repo just ooint to main branch.


Release: BYOL toolkit, models, and global mmlu-lite benchmark datasets