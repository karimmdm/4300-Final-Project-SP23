import requests
import os
import re

def mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def download_interests(data_dir):
    interests_dir = data_dir+"interests/"
    mkdir(interests_dir)
    
    interests = ["Realistic", "Investigative", "Artistic", "Social", "Enterprising", "Conventional"]
    
    params = {
        'fmt': 'csv',
    }
    
    for i in interests:
        url = "https://www.onetonline.org/explore/interests/{}/{}.csv".format(i, i)
        response = requests.get(url, params=params)

        with open(interests_dir + "{}.csv".format(i), 'wb') as f:
            f.write(response.content)

def download_values(data_dir):
    values_dir = data_dir+"values/"
    mkdir(values_dir)
    
    values = ["Achievement", "Independence", "Recognition", "Relationships", "Support", "Working_Conditions"]
    
    params = {
        'fmt': 'csv',
    }
    
    for v in values:
        url = "https://www.onetonline.org/explore/workvalues/{}/{}.csv".format(v, v)
        response = requests.get(url, params=params)

        with open(values_dir + "{}.csv".format(v), 'wb') as f:
            f.write(response.content)

def download_knowledge(data_dir):
    knowledge_dir = data_dir+"knowledge/"
    mkdir(knowledge_dir)

    def file_names(number, letters, page_names): #annoying path requries number an letter
        names =  [s.replace(" ", "_") for s in page_names]         
        suffix = [number] if len(page_names) == 1 else ["{}.{}".format(number, l) for l in letters]
        assert len(suffix) == len(page_names), (suffix, page_names)
        return list(zip(suffix, names))
    
    arts_hum = file_names(7, 'acbde', ["English Language", "Fine Arts", "Foreign Language", "History and Archeology", "Philosophy and Theology"])
    bus_man = file_names(1, 'abecfd', ["Administration and Management", "Administrative", "Customer and Personal Service", "Economics and Accounting",
                                       "Personnel and Human Resources", "Sales and Marketing"])
    comm = file_names(9, 'ba', ["Communcations and Media", "Telecommunications"])
    eng_tech = file_names(3, 'dacbe', ["Building and Construction", "Computers and Electronics", "Design",
                                       "Engineering and Technology", "Mechanical"])
    health_serv = file_names(5, 'ab', ["Medicine and Dentistry", "Therapy and Counseling"])
    law_pub = file_names(8, 'ba', ["Law and Government", "Public Safety and Security"])
    man_prod = file_names(2, 'ba', ["Food Production", "Production and Processing"])
    math_sci = file_names(4, 'dcgabef', ["Biology", "Chemistry", "Geography", "Mathematics", "Physics",
                                                "Psychology", "Sociology and Anthropology"])
    ed_train = file_names(6, '', ["Education and Training"])
    transp = file_names(10, '', ["Transportation"])
    knowledge = arts_hum + bus_man + comm + eng_tech + health_serv + law_pub + man_prod + math_sci + ed_train + transp
    
    params = {
        'fmt': 'csv',
    }

    for suffix, file_name in knowledge:
        url = "https://www.onetonline.org/find/descriptor/result/2.C.{}/{}.csv".format(suffix, file_name)
        response = requests.get(url, params=params)

        with open(knowledge_dir + "{}.csv".format(file_name), 'wb') as f:
            f.write(response.content)

def download_cross_skills(data_dir):
    cross_skills_dir = data_dir+"cross-skills/"
    mkdir(cross_skills_dir)
    skills_url = "https://www.onetonline.org/find/descriptor/browse/2.B/2.B.2/2.B.5/2.B.1/2.B.4/2.B.3"
    response = requests.get("https://www.onetonline.org/find/descriptor/browse/2.B/2.B.2/2.B.5/2.B.1/2.B.4/2.B.3")
    html_body = response.text
    
    skills_hrefs = re.findall(r"(?s)(?<=/find/descriptor/result/2.B.)(.*?)(?=</a>)", html_body)
    regex = (r"\">\n"
	r"    <span class=\"d-block text-center\" style=\"position: absolute; left: -2em; top: 0; width: 2em\"><i aria-hidden=\"true\" class=\"fas fa-list-alt\"></i></span>\n"
	r"    ")
    "\">\n<span class=\"d-block text-center\" style=\"position: absolute; left: -2em; top: 0; width: 2em\"><i aria-hidden=\"true\" class=\"fas fa-list-alt\"></i></span>"

    suffix_skill = [re.sub(regex, ",", s, 0, re.MULTILINE).split(",") for s in skills_hrefs]
    
    params = {
        'fmt': 'csv',
    }

    for suffix, file_name in suffix_skill:
        file_name = file_name.split("\n")[0]
        url = "https://www.onetonline.org/find/descriptor/result/2.B.{}/{}.csv".format(suffix, file_name)
        response = requests.get(url, params=params)

        with open(cross_skills_dir + "{}.csv".format(file_name), 'wb') as f:
            f.write(response.content)


if __name__ == "__main__":
    raw_data_dir = "raw_data/"
    mkdir(raw_data_dir)
    
    download_interests(raw_data_dir)
    download_knowledge(raw_data_dir)
    download_values(raw_data_dir)
    download_cross_skills(raw_data_dir)
    
    