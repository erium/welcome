import os


def test_correct_post_hook_prints():

    template_dirs = ['use_case_templates', 'workflow_templates']

    for template_dir in template_dirs:
        for template_subdir in os.listdir(template_dir):
            if not os.path.isdir(os.path.join(template_dir, template_subdir)):
                continue

            post_gen_script_path = os.path.join(
                template_dir, template_subdir, 'hooks', 'post_gen_project.py')
            print(post_gen_script_path)

            assert os.path.exists(post_gen_script_path)

            with open(post_gen_script_path, 'r', encoding='utf-8') as f:
                post_gen_code = f.readlines()

            # get last line with content
            last_line = None
            for line in post_gen_code[::-1]:
                line = line.strip()
                if not line.startswith('#'):
                    last_line = line
                    break

            assert last_line in ('print("Hal_Magic_Template_done.")',
                                 "print('Hal_Magic_Template_done.')")
