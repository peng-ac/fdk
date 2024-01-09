import pycurl
from time import gmtime, strftime

if __name__ == '__main__':
    c = pycurl.Curl()
    c.setopt(c.URL, 'https://hooks.slack.com/services/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')
    c.setopt(pycurl.CUSTOMREQUEST,"POST")
    # c.setopt(pycurl.POSTFIELDS,'{"text":"iFDK : Run Finished"}')
    stm = strftime("%Y-%m-%d %H:%M:%S", gmtime()) 
    sfields='{\"text\":\"iFDK : Run Finished at %s\"}' % (stm)
    print sfields
    c.setopt(pycurl.POSTFIELDS, sfields)
    c.perform()
    c.close()



