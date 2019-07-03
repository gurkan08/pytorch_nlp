# -*- coding: utf-8 -*-
"""
Created on Tue May 21 13:29:18 2019

@author: gurkan.sahin

flask web server olacak
girilen metnin sınıf değerini geri dönecek !
"""


from flask import Flask
from flask import url_for
from flask import redirect, render_template
from flask import request
from flask import abort

from Infer import Infer


app = Flask(__name__)



"""
.../infer directory ile başlangıç sayfasına git
"""
@app.route("/infer", methods = ["GET", "POST"])
def infer():	
    if request.method == "GET":
        return render_template("infer.html")
        
    if request.method == "POST":
        doc = request.form["metin"]
        pred_class = Infer.get_pred_class(doc)
        return render_template("infer.html", sonuc = pred_class)
    
    
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

    
    
