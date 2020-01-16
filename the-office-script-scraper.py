import requests
from bs4 import BeautifulSoup
import re

#num_episodes = [6, 22, 23, 14, 26, 24, 24, 24, 23]
num_episodes = [6, 22, 23, 14]

script = open('office_script.txt', "w+", encoding="utf-8")

for season in range(len(num_episodes)):
    for episode in range(num_episodes[season]):
        r = requests.get("http://officequotes.net/no{}-{}.php".format(season + 1, str(episode + 1).zfill(2) ) )
        soup = BeautifulSoup(r.text, "html.parser")

        print(soup.get_text())
        start = re.search('Season [0-9] - Episode ([0-9])+', soup.get_text()).start()

        if (re.search('Deleted Scene', soup.get_text()) != None):
            end = re.search('Deleted Scene', soup.get_text()).start()
        else:
            end = re.search(re.escape("""@officequotesnet
!function(d,s,id){var js,fjs=d.getElementsByTagName"""), soup.get_text()).start()

        print(soup.get_text()[start:end])

        script.write(soup.get_text()[start:end])
