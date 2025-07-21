import numpy as np

def extract_bracket(txt):
    first_index = txt.find("(")
    last_index = txt.rfind(")")

    if first_index != -1 and last_index != -1:
        first_index = first_index
        return txt[:first_index], txt[first_index+1:last_index]
    else: return txt, None

def un_bracket(txt, opener = '(', closer = ')'):
    first_index = txt.find(opener)
    last_index = txt.rfind(closer)
    if first_index == 0 and last_index == len(txt)-1:
        return txt[first_index+1:last_index]
    else: return txt

def brackets_to_argv(txt):
    return txt.split(",")

def equality_split(txt, equal = '='):
    sections = txt.split(equal)

    if len(sections) > 1:
        return sections[0],sections[1]
    else: return sections[0], None

def brackets_to_kwargs(txt, divider = ',', equal= '=',):
    ret = {}
    sections = txt.split(divider)

    for section in sections:
        s = section.split(equal)
        if len(s) > 1:
            ret[s[0]] = s[1]
        else:
            ret[s[0]] = s[0]

    return ret

def txt_to_sections(txt,opener = '(', closer = ')', divider = ',', equal= '=', check_type = False):
    #print(txt)
    indices = []
    opened = 0

    is_array = 1 #0 - means it is an object, 1 - means it is an array
    #if there is no "=" outside the brackets then it is an array
    #SkLearnModel(...), SkLearnModel(model(mlp),hls(100,100,100) - array
    # model=mlp,hls=(100,100,100 = object

    for i, sign in enumerate(txt):
        #print(txt[i], divider, txt[i] == divider)
        if txt[i] == opener:
            opened = opened +1
        elif txt[i] == closer:
            opened = opened - 1
        elif txt[i] == equal and check_type:
            #print(txt[i], i, opened)
            if opened == 0:
                is_array = 0
        elif txt[i] == divider:
            if opened == 0:
                indices.append(i)

    #print(indices)
    ret = None
    if len(indices) > 0:
        ret = [txt[i+1:j] for i,j in zip([-1] + indices, indices[0:]+[len(txt)])]
    else: ret = [txt]


    #print(ret, is_array)
    if check_type:
        return ret, is_array
    else:
        return ret

def text_to_json_array(txt, opener = '(', closer = ')', divider = ',', equal= '=', depth=0):

    ret = txt
    openers = txt.count(opener)
    if openers != txt.count(closer):
        raise ValueError("Inconsistent brackets in text {}".format(txt))

    #types:
    #a,b,c
    #a(),b(),c()
    #a=z,b=g,c=y #object 100%
    sections, semantic_type = txt_to_sections(txt,opener,closer,divider, check_type=True)

    #print(depth * " ", txt, sections, semantic_type)

    #no brackets, no commas, no equals
    if openers == 0 and len(sections) == 1 and txt.count(equal) == 0:
        return ret

    if semantic_type == 1:
        ret = []
    else:
        ret = {}

    #print(depth * " ", sections, semantic_type)
    # a,b,c
    # a(),b(),c()
    if semantic_type == 1:
        for section in sections:
            param_name, param_bracket = extract_bracket(section)
            # a,b,c
            if param_bracket is None:
                #print(depth * " ", param_name, "a,b,c")
                ret.append(text_to_json_array(param_name,opener, closer, divider,equal, depth+1))
            # a(),b(),c()
            else:
                #print(depth * " ", txt, "a(),b(),c()")
                #print(depth * " ", param_name, param_bracket)
                obj = text_to_json_array(param_bracket, opener, closer, divider, equal, depth+1)
                #if obj == "": obj = {} #for empty string
                ret.append({param_name: obj})
    # a=z,b=g,c=y()
    else:
        for section in sections:
            param = txt_to_sections(section, opener = '(', closer = ')', divider = '=')
            print(depth * " ", section,  "a=z,b=g,c=y()")
            print(depth * " ", "param", param)
            ret[param[0]] = text_to_json_array(un_bracket(param[1], opener, closer),opener, closer, divider,equal, depth+1)

    # end - if no brackets and equals
    return ret

def module_loader_json(sys_modules, prefix, models):
    models_classes = {}
    models_instances = []
    # before = [str(m) for m in sys.modules]
    # after = [str(m) for m in sys.modules]
    # mods = [m for m in after if not m in before]

    for model in models:
        model_name = None
        args = None
        if isinstance(model, str):
            model_name = model
        else:
            for m in model:
                model_name = m
                args = model[m]

        #print("model", model)
        #print("args", args)
        # currently unused
        model_txt = prefix + "." + model_name
        #print(model_txt)
        #print(sys_modules[model_txt], model_name)
        models_classes[model_name] = getattr(sys_modules[model_txt], model_name)
        # self._models_classes[model_name] = sys.modules[model_txt]

        # create a class instance
        if args:
            models_instances.append(getattr(sys_modules[model_txt], model_name)(args))
        else:
            models_instances.append(getattr(sys_modules[model_txt], model_name)())

    print("Loaded models: \n", models_classes)
    print("Instances created: \n", models_instances)

    return models_classes, models_instances


def module_loader(sys_modules, prefix, models_str):
    models = text_to_json_array(models_str)

    print("model_str", models_str)
    print("json", models)

    #for only one simple model
    if isinstance(models, str):
        models = [models]

    return module_loader_json(sys_modules, prefix, models)


def standard_storage_init():
    pass

from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

def bias_metric(y_true, y_pred):
    return abs(np.mean(y_true) - np.mean(y_pred))

def power_metric(y_true, y_pred):
    return abs(len(y_true)*5/60 * (np.mean(y_true) - np.mean(y_pred)))

def SMAPE(y_true, y_pred):
    return 0.01 / len(y_true) * np.sum([np.abs((p-t)/((p+t)*0.5)) for p,t in zip(y_pred, y_true)])

def oMAPE(y_true, y_pred):
    y_true[y_true == 0] = 0.001 # small value instead of zero
    ret = 0.01 / len(y_true) * np.sum([abs((p-t)/t) for p,t in zip(y_pred, y_true)])
    return ret

def metric_selection(metric_txt):
    metric = []
    for m in metric_txt:
        if m == "R^2":
            metric.append(r2_score)
        elif m == "MAE":
            metric.append(mean_absolute_error)
        elif m == "MSE":
            metric.append(mean_squared_error)
        elif m == "MAPE":
            metric.append(oMAPE)
        elif m == "BIAS":
            metric.append(bias_metric)
        elif m == "POW":
            metric.append(power_metric)
        else:
            raise ValueError("Not known metric " + m)

    return metric

import time
class MyTimer():
    def __init__(self):
        self._start_time = time.time()

    def get_time(self):
        return time.time() - self._start_time

    def make_interval(self):
        st = self._start_time
        self._start_time = time.time()
        return self._start_time - st

from threading import Timer

class RepeatedTimer(object):
    def __init__(self, interval, function, *args, **kwargs):
        self._timer = None
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.is_running = False
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False

