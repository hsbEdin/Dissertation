import click
from pm_parser import PmParser
from gensim_w2v import W2V

@click.group()
def cli():
    pass

@cli.command()
@click.option('--output-folder', default="/", help='folder for storing results')
@click.option('--name', default="", help='file name template')
@click.option('--log-file', default="parse.log", help='log file name')
@click.argument('input-file')
def parse_file(output_folder, name, log_file, input_file):
    print("Parsing file:", input_file)
    PmParser(log_file).parse_file(input_file, output_folder, name)

@cli.command()
@click.option('--output-folder', default="", help='folder for storing results')
@click.option('--ratio', default=0, help='num input files parsed into 1 output file(default: all files into 1 file)')
@click.option('--start-file', default=None, help='file from which to begin parsing')
@click.option('--end-file', default=None, help='file at which to end parsing')
@click.option('--name-count', default=1, help='output file name counter initial value')
@click.option('--ext', default="xml.gz", help='file extension to look for in folder')
@click.option('--name', default="", help='file name template')
@click.option('--log-file', default="parse.log", help='log file name')
@click.argument('input-folder')
def parse_folder(output_folder, ratio, start_file, end_file, name_count, ext, name, log_file, input_folder):
    PmParser(log_file).parse_folder(output_folder, ratio, start_file, end_file, name_count, ext, name, input_folder)

@cli.command()
@click.argument('file-path')
@click.option('--log-file', default="parse.log", help='log file name')
def load(file_path, log_file):
    print("Loading file:", file_path)
    PmParser(log_file).load(file_path)

@cli.command()
@click.option('--output-folder', default="", help='folder for storing results')
@click.option('--ratio', default=0, help='num input files parsed into 1 output file(default: all files into 1 file)')
@click.option('--keyword', default=None, help='word phrase to filter articles by, if none given all articles converted to text')
@click.option('--start-file', default=None, help='file from which to begin parsing')
@click.option('--end-file', default=None, help='file at which to end parsing')
@click.option('--name-count', default=1, help='output file name counter initial value')
@click.option('--ext', default="xml.gz", help='file extension to look for in folder')
@click.option('--name', default="", help='file name template')
@click.option('--log-file', default="parse.log", help='log file name')
@click.argument('input-folder')
def parse_folder_to_text(output_folder, ratio, keyword, start_file, end_file, name_count, ext, name, log_file, input_folder):
    PmParser(log_file).parse_folder_to_text(output_folder, ratio, keyword, start_file, end_file, name_count, ext, name, input_folder)

@cli.command()
@click.option('--output-folder', default="/", help='folder for storing results')
@click.option('--output-file', default=None, help='name of model file')
@click.option('--preprop', default=0, help='preprocessing level: 0-none,1-gensim simple preproc,2-stopword removal,3-stopword removal+lowercasting,4-3+stemming')
@click.option('--size', default=150, help='size of resulting vectors')
@click.option('--window', default=5, help='window size')
@click.option('--min-count', default=1, help='min count')
@click.option('--workers', default=4, help='number of workers used for training')
@click.option('--log-file', default="parse.log", help='log file name')
@click.option('--atype', default=1, help='0 is CBOW, 1 is Skipgram')
@click.option('--start-file', default=None, help='file from which to begin parsing')
@click.option('--end-file', default=None, help='file at which to end parsing')
@click.option('--ext', default="txt.gz", help='file extension to look for in folder')
@click.option('--input-file', default=None, help='input file')
@click.option('--gram', default='word', help="'word' for simple word embedding, 'phrase' for more complex phrase embedding")
@click.argument('input-folder')
def train_w2v(output_folder, output_file, preprop, size, window, min_count, workers, log_file, atype, input_folder, start_file, end_file, ext, input_file, gram):
    W2V(log_file).train(input_folder, output_folder, output_file, preprop, size, window, min_count, workers, atype, start_file, end_file, ext, input_file, gram)

@cli.command()
@click.option('--output-folder', default="/", help='folder for storing results')
@click.option('--output-file', default=None, help='name of model file')
@click.option('--preprop', default=0, help='preprocessing level: 0-none,1-gensim simple preproc,2-stopword removal,3-stopword removal+lowercasting,4-3+stemming')
@click.option('--size', default=150, help='size of resulting vectors')
@click.option('--window', default=5, help='window size')
@click.option('--min-count', default=1, help='min count')
@click.option('--workers', default=4, help='number of workers used for training')
@click.option('--log-file', default="parse.log", help='log file name')
@click.option('--atype', default=1, help='0 is CBOW, 1 is Skipgram')
@click.option('--start-file', default=None, help='file from which to begin parsing')
@click.option('--end-file', default=None, help='file at which to end parsing')
@click.option('--ext', default="txt.gz", help='file extension to look for in folder')
@click.option('--input-file', default=None, help='input file')
@click.option('--gram', default='word', help="'word' for simple word embedding, 'phrase' for more complex phrase embedding")
@click.argument('input-folder')
@click.argument('pretrained-model')
def intersect(pretrained_model, output_folder, output_file, preprop, size, window, min_count, workers, log_file, atype, input_folder, start_file, end_file, ext, input_file, gram):
    W2V(log_file).intersect(pretrained_model, input_folder, output_folder, output_file, preprop, size, window, min_count, workers, log_file, atype, start_file, end_file, ext, input_file, gram)

@cli.command()
@click.argument('input-file')
@click.argument('output-file')
@click.option('--log-file', default="parse.log", help='log file name')
def bin2txt(input_file, output_file, log_file):
    W2V(log_file).bin2text(input_file, output_file)

@cli.command()
@click.argument('input_folder')
@click.argument('output-file')
@click.option('--start-file', default=None, help='file from which to begin parsing')
@click.option('--end-file', default=None, help='file at which to end parsing')
@click.option('--ext', default="txt.gz", help='file extension to look for in folder')
@click.option('--log-file', default="parse.log", help='log file name')
def merge_txt_files(input_folder, output_file, start_file, end_file, ext, log_file):
    PmParser(log_file).merge_txt_files(input_folder, output_file, start_file, end_file, ext)

if __name__ == '__main__':
    cli()