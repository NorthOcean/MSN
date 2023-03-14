"""
@Author: Conghao Wong
@Date: 2021-08-05 15:26:57
@LastEditors: Conghao Wong
@LastEditTime: 2022-11-14 11:38:58
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import re
import os
import sys

sys.path.insert(0, os.path.abspath('.'))

from silverballers.__args import SilverballersArgs, AgentArgs, HandlerArgs
from codes.args import Args


FLAG = '<!-- DO NOT CHANGE THIS LINE -->'
TARGET_FILE = './README.md'
MAX_SPACE = 20


def read_comments(args: Args) -> list[str]:

    results = []
    for arg in args._arg_list:

        name = arg
        default = args._args_default[name]
        dtype = type(default).__name__
        argtype = args._arg_type[name]

        short_name_desc = ''
        if name in args._arg_short_name.values():
            short_names = []
            for key in args._arg_short_name.keys():
                if args._arg_short_name[key] == name:
                    short_names.append(key)

            ss = ' '.join(['`-{}`'.format(s) for s in short_names])
            short_name_desc = f' (short for {ss})'

        doc = getattr(args.__class__, arg).__doc__
        doc = doc.replace('\n', ' ')
        for _ in range(MAX_SPACE):
            doc = doc.replace('  ', ' ')

        s = (f'- `--{name}`' + short_name_desc +
             f': type=`{dtype}`, argtype=`{argtype}`.\n' +
             f' {doc}\n  The default value is `{default}`.')
        results.append(s + '\n')
        # print(s)

    return results


def get_doc(args: list[Args], titles: list[str]) -> list[str]:

    new_lines = []
    all_args = []

    for arg, title in zip(args, titles):
        new_lines += [f'\n### {title}\n\n']
        c = read_comments(arg)
        c.sort()

        for new_line in c:
            name = new_line.split('`')[1]
            if name not in all_args:
                all_args.append(name)
                new_lines.append(new_line)

    return new_lines


def update_readme(new_lines: list[str], md_file: str):
    with open(md_file, 'r') as f:
        lines = f.readlines()
    lines = ''.join(lines)

    try:
        pattern = re.findall(
            f'([\s\S]*)({FLAG})([\s\S]*)({FLAG})([\s\S]*)', lines)[0]
        all_lines = list(pattern[:2]) + new_lines + list(pattern[-2:])

    except:
        flag_line = f'{FLAG}\n'
        all_lines = [lines, flag_line] + new_lines + [flag_line]

    with open(md_file, 'w+') as f:
        f.writelines(all_lines)


def print_help_info(value: str):

    from codes.args import Args
    from silverballers.__args import SilverballersArgs, AgentArgs, HandlerArgs
    from scripts.update_readme import get_doc

    files = [Args(is_temporary=True),
             SilverballersArgs(is_temporary=True),
             AgentArgs(is_temporary=True),
             HandlerArgs(is_temporary=True)]

    titles = ['Basic args',
              'Silverballers args',
              'First-stage silverballers args',
              'Second-stage silverballers args']

    doc_lines = get_doc(files, titles)
    if value == 'all_args':
        [print(doc) for doc in doc_lines]
    else:
        doc_lines = [doc for doc in doc_lines if doc[5:].startswith(value)]
        [print(doc) for doc in doc_lines]


if __name__ == '__main__':
    files = [Args(is_temporary=True),
             SilverballersArgs(is_temporary=True),
             AgentArgs(is_temporary=True),
             HandlerArgs(is_temporary=True)]

    titles = ['Basic args',
              'Silverballers args',
              'First-stage silverballers args',
              'Second-stage silverballers args']

    doc = get_doc(files, titles)
    update_readme(doc, TARGET_FILE)
