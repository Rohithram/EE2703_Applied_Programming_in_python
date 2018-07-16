c = get_config()
c.Exporter.preprocessors = [ 'bibpreprocessor.BibTexPreprocessor', 'pymdpreprocessor.PyMarkdownPreprocessor' ]
c.Exporter.template_file = 'report_nocode.tplx'
