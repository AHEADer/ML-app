<div class="container">

    <form class="well form-horizontal" action="/dataset1" method="post"  id="contact_form">
<fieldset>

<!-- Form Name -->
<legend>Test if you have a heart disease!</legend>

<!-- Text input-->

<div class="form-group">
  <label class="col-md-4 control-label">Age</label>  
  <div class="col-md-4 inputGroupContainer">
  <div class="input-group">
  <span class="input-group-addon"><i class="glyphicon glyphicon-user"></i></span>
  <input  name="age" placeholder="Age" class="form-control"  type="text">
    </div>
  </div>
</div>

<!-- Select Basic -->
<div class="form-group">
  <label class="col-md-4 control-label">Sex</label>
    <div class="col-md-4 selectContainer">
    <div class="input-group">
        <span class="input-group-addon"><i class="glyphicon glyphicon-list"></i></span>
    <select name="sex" class="form-control selectpicker" >
      <option value=" " >Please select your sex</option>
      <option value="1">Male</option>
      <option value="0">Female</option>
    </select>
  </div>
</div>
</div>
  
<!-- Select Basic -->
<div class="form-group">
  <label class="col-md-4 control-label">Chest pain type</label>
    <div class="col-md-4 selectContainer">
    <div class="input-group">
        <span class="input-group-addon"><i class="glyphicon glyphicon-list"></i></span>
    <select name="cp" class="form-control selectpicker" >
      <option value=" " >Please select your chest pain type</option>
      <option value="1">typical angina</option>
      <option value="2">atypical angina</option>
      <option value="3">non-angina</option>
      <option value="4">asymptomatic angina</option>
    </select>
  </div>
</div>
</div>

<!-- Text input-->
<div class="form-group">
  <label class="col-md-4 control-label">resting blood pressure on admission to hospital</label>  
    <div class="col-md-4 inputGroupContainer">
    <div class="input-group">
        <span class="input-group-addon"><i class="glyphicon glyphicon-envelope"></i></span>
  <input name="restbp" placeholder="continuous (mmHg)" class="form-control"  type="text">
    </div>
  </div>
</div>

<!-- Text input-->
<div class="form-group">
  <label class="col-md-4 control-label">serum cholesterol level</label>  
    <div class="col-md-4 inputGroupContainer">
    <div class="input-group">
        <span class="input-group-addon"><i class="glyphicon glyphicon-envelope"></i></span>
  <input name="chol" placeholder="continuous (mg/dl)" class="form-control"  type="text">
    </div>
  </div>
</div>

<!-- Select Basic -->
<div class="form-group">
  <label class="col-md-4 control-label">fasting blood sugar</label>
    <div class="col-md-4 selectContainer">
    <div class="input-group">
        <span class="input-group-addon"><i class="glyphicon glyphicon-list"></i></span>
    <select name="fbs" class="form-control selectpicker" >
      <option value=" " >Please select your fbs</option>
      <option value="0"><= 120 mg/dl</option>
      <option value="1">> 120 mg/dl</option>
    </select>
  </div>
</div>
</div>

<!-- Select Basic -->
<div class="form-group">
  <label class="col-md-4 control-label">resting electrocardiography</label>
    <div class="col-md-4 selectContainer">
    <div class="input-group">
        <span class="input-group-addon"><i class="glyphicon glyphicon-list"></i></span>
    <select name="restecg" class="form-control selectpicker" >
      <option value=" " >Please select your restecg</option>
      <option value="0">normal</option>
      <option value="1">ST-T wave abnormality</option>
      <option value="2">left ventricular hypertrophy</option>
    </select>
  </div>
</div>
</div>

<!-- Text input-->
<div class="form-group">
  <label class="col-md-4 control-label">maximum heart rate achieved</label>  
    <div class="col-md-4 inputGroupContainer">
    <div class="input-group">
        <span class="input-group-addon"><i class="glyphicon glyphicon-envelope"></i></span>
  <input name="thalach" placeholder="maximum heart rate achieved" class="form-control"  type="text">
    </div>
  </div>
</div>

<!-- Select Basic -->
<div class="form-group">
  <label class="col-md-4 control-label">exercise induced angina</label>
    <div class="col-md-4 selectContainer">
    <div class="input-group">
        <span class="input-group-addon"><i class="glyphicon glyphicon-list"></i></span>
    <select name="exang" class="form-control selectpicker" >
      <option value=" " >Please select your exercise induced angina</option>
      <option value="0">no</option>
      <option value="1">yes</option>
    </select>
  </div>
</div>
</div>

<!-- Text input-->
<div class="form-group">
  <label class="col-md-4 control-label">ST depression induced by exercise relative to rest</label>  
    <div class="col-md-4 inputGroupContainer">
    <div class="input-group">
        <span class="input-group-addon"><i class="glyphicon glyphicon-envelope"></i></span>
  <input name="oldpeak" placeholder="continuous" class="form-control"  type="text">
    </div>
  </div>
</div>

<!-- Select Basic -->
<div class="form-group">
  <label class="col-md-4 control-label">slope of peak exercise ST segment</label>
    <div class="col-md-4 selectContainer">
    <div class="input-group">
        <span class="input-group-addon"><i class="glyphicon glyphicon-list"></i></span>
    <select name="slope" class="form-control selectpicker" >
      <option value=" " >Please select your slope</option>
      <option value="1">upsloping</option>
      <option value="2">flat</option>
      <option value="3">downsloping</option>
    </select>
  </div>
</div>
</div>

<!-- Select Basic -->
<div class="form-group">
  <label class="col-md-4 control-label">number of major vessels colored by fluoroscopy</label>
    <div class="col-md-4 selectContainer">
    <div class="input-group">
        <span class="input-group-addon"><i class="glyphicon glyphicon-list"></i></span>
    <select name="ca" class="form-control selectpicker" >
      <option value=" " >Please select your number of major vessels colored by fluoroscopy</option>
      <option value="0">0</option>
      <option value="1">1</option>
      <option value="2">2</option>
      <option value="3">3</option>
    </select>
  </div>
</div>
</div>

<!-- Select Basic -->
<div class="form-group">
  <label class="col-md-4 control-label">categorical</label>
    <div class="col-md-4 selectContainer">
    <div class="input-group">
        <span class="input-group-addon"><i class="glyphicon glyphicon-list"></i></span>
    <select name="thal" class="form-control selectpicker" >
      <option value=" " >Please select your </option>
      <option value="3">normal</option>
      <option value="6">fixed defect</option>
      <option value="7">reversible defect</option>
    </select>
  </div>
</div>
</div>

<!-- Select Basic
<div class="form-group">
  <label class="col-md-4 control-label">diagnosis of heart disease</label>
    <div class="col-md-4 selectContainer">
    <div class="input-group">
        <span class="input-group-addon"><i class="glyphicon glyphicon-list"></i></span>
    <select name="num" class="form-control selectpicker" >
      <option value=" " >Please select your num</option>
      <option value="0">less than 50% narrowing in any major vessel</option>
      <option value="1">1 that has 50% narrowing in 1-4 vessels</option>
      <option value="2">2 that have 50% narrowing in 1-4 vessels</option>
      <option value="3">3 that have 50% narrowing in 1-4 vessels</option>
      <option value="4">4 that have 50% narrowing in 1-4 vessels</option>
    </select>
  </div>
</div>
</div> -->
<!-- Above are what I modified-->

<!-- Select Basic -->
<div class="form-group">
  <label class="col-md-4 control-label">Select your model</label>
    <div class="col-md-4 selectContainer">
    <div class="input-group">
        <span class="input-group-addon"><i class="glyphicon glyphicon-list"></i></span>
    <select name="model" class="form-control selectpicker" >
      <option value=" " >Please select your model</option>
      <option value="svm">SVM</option>
      <option value="naive">Naive Bayes</option>
      <option value="logistic">Logistic Regression</option>
      <option value="boost">Boost gradient</option>
    </select>
  </div>
</div>
</div>

<!-- Success message -->
<div class="alert alert-success" role="alert" id="success_message">= = <i class="glyphicon glyphicon-thumbs-up"></i> : )</div>

<!-- Button -->
<div class="form-group">
  <label class="col-md-4 control-label"></label>
  <div class="col-md-4">
    <button type="submit" class="btn btn-warning" >Send <span class="glyphicon glyphicon-send"></span></button>
  </div>
</div>

</fieldset>
</form>
</div>
    </div><!-- /.container --> 