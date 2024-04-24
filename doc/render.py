"""
The function of this script is to render the Jinja2 templates in the current
directory into input files for sphinx. It introspects the OpenMM-ML Python module
to find all of the classes and formats them for inclusion into the templates.
"""
from os.path import dirname, join, splitext, basename
from glob import glob
import inspect

import jinja2
import openmmml



def fullname(klass):
    return klass.__module__ + '.' + klass.__name__


def models_template_variables():
    """Create the data structure available to the Jinja2 renderer when
    filling in the templates.

    This function extracts all of classes in ``openmmml.models`` and returns
    a list of them
    """
    data = {
        'models': [],
    }

    for _, module in inspect.getmembers(openmmml.models, predicate=inspect.ismodule):
        for name, obj in inspect.getmembers(module, predicate=inspect.isclass):
            if issubclass(obj, openmmml.mlpotential.MLPotentialImpl) and obj != openmmml.mlpotential.MLPotentialImpl:
                data['models'].append(fullname(obj))

    return data


def main():
    here = dirname(__file__)
    templateLoader = jinja2.FileSystemLoader(here)
    templateEnv = jinja2.Environment(loader=templateLoader)
    data = models_template_variables()

    for template_fn in map(basename, glob(join(here, '*.jinja2'))):
        output_fn = splitext(template_fn)[0]
        print('Rendering %s to %s...' % (template_fn, output_fn))
        template = templateEnv.get_template(template_fn)
        output_text = template.render(data)
        with open(output_fn, 'w') as f:
            f.write(output_text)


if __name__ == '__main__':
    main()