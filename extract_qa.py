with open('app/main.py') as f:
    content = f.read()
idx = content.find('_extractive_qa')
if idx >= 0:
    section = content[idx:]
    print(section[:3000])