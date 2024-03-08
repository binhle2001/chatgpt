# 1. Prepare data
## Create a txt file following this format:
```
Copy code
  Title: Your product,
  URL: Your link to product/documentation,
  -----
  **Your descriptions for your chatbot**
  
  * * *
  # Domain 1:
    Details for domain 1
  # Domain 2:
    Details for domain 2
```
## Please follow the example in the file Account_Setting.txt
# 2. Prepare virtual environments
follow code:
```
 python -m venv .venv
 pip install -r requirements.txt
```
# 3. Run code:
```
python embedding.py
uvicorn main:app --host 0.0.0.0 --port your_port --reload
```
# 4. Call API
- Example: Call API 
