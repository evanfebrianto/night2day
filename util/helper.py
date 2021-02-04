from IPython.core.display import display, HTML

def show_training_result(name=None):
    display(HTML('./checkpoints/{}/web/index.html'.format(name)))

def show_testing_result(name=None, epoch=0):
    display(HTML('./results/{}/test_{}/index.html'.format(name, str(epoch))))