#!/usr/bin/env python
#
# corenlp  - Python interface to Stanford Core NLP tools
# Copyright (c) 2012 Dustin Smith
#   https://github.com/dasmith/stanford-corenlp-python
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.


import json
import optparse
import os
import re
import sys
import traceback
import pexpect
import tempfile
import shutil
from progressbar import ProgressBar, Fraction
from unidecode import unidecode
from subprocess import call

VERBOSE = False
#VERBOSE = True
STATE_START, STATE_TEXT, STATE_WORDS, STATE_TREE, STATE_DEPENDENCY, STATE_COREFERENCE = 0, 1, 2, 3, 4, 5
WORD_PATTERN = re.compile('\[([^\]]+)\]')
CR_PATTERN = re.compile(r"\((\d*),(\d)*,\[(\d*),(\d*)\)\) -> \((\d*),(\d)*,\[(\d*),(\d*)\)\), that is: \"(.*)\" -> \"(.*)\"")

DIRECTORY = "/usr/local/stanford-corenlp-full-2014-06-16/"

class bc:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

class ProcessError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

class ParserError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

class TimeoutError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class OutOfMemoryError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def init_corenlp_command(corenlp_path, memory, properties,
                         classname="edu.stanford.nlp.pipeline.StanfordCoreNLP"):
    """
    Checks the location of the jar files.
    Spawns the server as a process.
    """

    # TODO: Can edit jar constants
#    jars = ["stanford-corenlp-3.2.0.jar",
#            "stanford-corenlp-3.2.0-models.jar",
    jars = ["stanford-corenlp-3.4.jar",
            "stanford-corenlp-3.4-models.jar",            
            "xom.jar",
            "joda-time.jar",
            "ejml-0.23.jar",
            "jollyday.jar"
            ]

    java_path = "java"

    # use properties file or, in case of Sentiment analysis, something else
    if classname == "edu.stanford.nlp.pipeline.StanfordCoreNLP":
        # include the properties file, so you can change defaults
        # but any changes in output format will break parse_parser_results()
        current_dir_pr =  os.path.dirname(os.path.abspath(__file__)
                                          ) + "/" + properties
        if os.path.exists(properties):
            props = "-props %s" % (properties.replace(" ", "\\ "))
        elif os.path.exists(current_dir_pr):
            props = "-props %s" % (current_dir_pr.replace(" ", "\\ "))
        else:
            raise Exception("Error! Cannot locate: %s" % properties)
    else:
        props = '-output PROBABILITIES,ROOT -stdin'

    # add and check classpaths
    jars = [corenlp_path + "/" + jar for jar in jars]
    for jar in jars:
        if not os.path.exists(jar):
            raise Exception("Error! Cannot locate: %s" % jar)

    # add memory limit on JVM
    if memory:
        limit = "-Xmx%s" % memory
    else:
        limit = ""

    return "%s %s -cp %s %s %s" % (java_path, limit, ':'.join(jars), classname, props)


def remove_id(word):
    """Removes the numeric suffix from the parsed recognized words: e.g. 'word-2' > 'word' """
    return word.replace("'", "")


def parse_bracketed(s):
    '''Parse word features [abc=... def = ...]
    Also manages to parse out features that have XML within them
    '''
    word = None
    attrs = {}
    temp = {}
    # Substitute XML tags, to replace them later
    for i, tag in enumerate(re.findall(r"(<[^<>]+>.*<\/[^<>]+>)", s)):
        temp["^^^%d^^^" % i] = tag
        s = s.replace(tag, "^^^%d^^^" % i)
    # Load key-value pairs, substituting as necessary
    for attr, val in re.findall(r"([^=\s]*)=([^=\s]*)", s):
        if val in temp:
            val = temp[val]
        if attr == 'Text':
            word = val
        else:
            attrs[attr] = val
    return (word, attrs)


def parse_parser_results(text):
    """ This is the nasty bit of code to interact with the command-line
    interface of the CoreNLP tools.  Takes a string of the parser results
    and then returns a Python list of dictionaries, one for each parsed
    sentence.
    """
    results = {"sentences": []}
    state = STATE_START
    for line in unidecode(text.decode('utf-8')).split("\n"):
        line = line.strip()

        if line.startswith("Sentence #"):
            sentence = {'words': [], 'parsetree': [], 'dependencies': []}
            results["sentences"].append(sentence)
            state = STATE_TEXT

        elif state == STATE_TEXT:
            sentence['text'] = line
            state = STATE_WORDS

        elif state == STATE_WORDS:
            if not line.startswith("[Text="):
                raise ParserError('Parse error. Could not find "[Text=" in: %s' % line)
            for s in WORD_PATTERN.findall(line):
                sentence['words'].append(parse_bracketed(s))
            state = STATE_TREE

        elif state == STATE_TREE:
            if len(line) == 0:
                state = STATE_DEPENDENCY
                sentence['parsetree'] = " ".join(sentence['parsetree'])
            else:
                sentence['parsetree'].append(line)

        elif state == STATE_DEPENDENCY:
            if len(line) == 0:
                state = STATE_COREFERENCE
            else:
                split_entry = re.split("\(|, |-", line[:-1])
                if len(split_entry) == 5:
                    rel, left, leftindex, right, rightindex = map(lambda x: remove_id(x), split_entry)
                    sentence['dependencies'].append(tuple([rel, left, leftindex, right, rightindex]))

        elif state == STATE_COREFERENCE:
            if "Coreference set" in line:
                if 'coref' not in results:
                    results['coref'] = []
                coref_set = []
                results['coref'].append(coref_set)
            else:
                for src_i, src_pos, src_l, src_r, sink_i, sink_pos, sink_l, sink_r, src_word, sink_word in CR_PATTERN.findall(line):
                    src_i, src_pos, src_l, src_r = int(src_i) - 1, int(src_pos) - 1, int(src_l) - 1, int(src_r) - 1
                    sink_i, sink_pos, sink_l, sink_r = int(sink_i) - 1, int(sink_pos) - 1, int(sink_l) - 1, int(sink_r) - 1
                    coref_set.append(((src_word, src_i, src_pos, src_l, src_r), (sink_word, sink_i, sink_pos, sink_l, sink_r)))

    return results

def parse_parser_sentimentLines(text):
    """ similar nasty to parse_parser_results, but designed to parse the
        kind of lines that are returned from the sentiment analysis code
        in the interactive mode
    """
    results = dict()
    for i,line in enumerate(unidecode(text.decode('utf-8')).split("\n")):
        line = re.sub(r'\r','',line)
#        print i,'"%s"' % line
        if (i <= 1):
            continue  # 1st line is just the input sentence, skip
        elif (i == 2):
            results['tree'] = line  # the PENNTREE output
            continue
        elif (i == 3):
            results['nodeinfo'] = list()    # other stuff like PROBABILITIES
            results['nodeinfo'].append(line)
        else:
            results['nodeinfo'].append(line)

    return results


def parse_parser_xml_results(xml, file_name="", raw_output=False):
    import xmltodict
    from collections import OrderedDict

    def extract_words_from_xml(sent_node):
        exted = map(lambda x: x['word'], sent_node['tokens']['token'])
        return exted

    # Turning the raw xml into a raw python dictionary:
    raw_dict = xmltodict.parse(xml)
    if raw_output:
        return raw_dict

    document = raw_dict[u'root'][u'document']

    # Making a raw sentence list of dictionaries:
    raw_sent_list = document[u'sentences'][u'sentence']

    # Convert sentences to the format like python
    # TODO: If there is only one sentence in input sentence,
    # raw_sent_list is dict and cannot decode following code...
    sentences = [{'dependencies': [[dep['dep'][i]['@type'],
                                    dep['dep'][i]['governor']['#text'],
                                    dep['dep'][i]['governor']['@idx'],
                                    dep['dep'][i]['dependent']['#text'],
                                    dep['dep'][i]['dependent']['@idx']]
                                   for dep in raw_sent_list[j][u'dependencies']
                                   if 'dep' in dep
                                   for i in xrange(len(dep['dep']))
                                   if dep['@type'] == 'collapsed-ccprocessed-dependencies'],
                  'sentimentValue': str(raw_sent_list[j]['@sentimentValue']),
                  'sentiment': str(raw_sent_list[j]['@sentiment']),
                  'text': extract_words_from_xml(raw_sent_list[j]),
                  'parsetree': str(raw_sent_list[j]['parse']),
                  'words': [[str(token['word']), OrderedDict([
                      ('CharacterOffsetEnd', str(token['CharacterOffsetEnd'])),
                      ('CharacterOffsetBegin', str(token['CharacterOffsetBegin'])),
                      ('PartOfSpeech', str(token['POS'])),
                      ('Lemma', str(token['lemma']))])]
                  for index, token in enumerate(raw_sent_list[j][u'tokens'][u'token'])]}

                 for j in xrange(len(raw_sent_list))]


    results = {'sentences': sentences}

    if file_name:
        results['file_name'] = file_name

    return results


def parse_xml_output(input_dir, corenlp_path=DIRECTORY, memory="3g",
                     raw_output=False, properties='default.properties'):
    """Because interaction with the command-line interface of the CoreNLP
    tools is limited to very short text bits, it is necessary to parse xml
    output"""
    #First, we change to the directory where we place the xml files from the
    #parser:

    xml_dir = tempfile.mkdtemp()
    file_list = tempfile.NamedTemporaryFile()

    #we get a list of the cleaned files that we want to parse:

    files = [input_dir + '/' + f for f in os.listdir(input_dir) if f.endswith(".txt")]

    #creating the file list of files to parse

    file_list.write('\n'.join(files))
    file_list.seek(0)

    command = init_corenlp_command(corenlp_path, memory, properties)\
        + ' -filelist %s -outputDirectory %s' % (file_list.name, xml_dir)

    #creates the xml file of parser output:

    call(command, shell=True)

    #reading in the raw xml file:
    # result = []
    try:
        for output_file in os.listdir(xml_dir):
            with open(xml_dir + '/' + output_file, 'r') as xml:
                # parsed = xml.read()
                file_name = re.sub('.xml$', '', os.path.basename(output_file))
                # result.append(parse_parser_xml_results(xml.read(), file_name,
                #                                        raw_output=raw_output))
                yield parse_parser_xml_results(xml.read(), file_name,
                                               raw_output=raw_output)
    finally:
        file_list.close()
        shutil.rmtree(xml_dir)
    # return result


class StanfordCoreNLP:

    """
    Command-line interaction with Stanford's CoreNLP java utilities.
    Can be run as a JSON-RPC server or imported as a module.
    """

    def _spawn_corenlp(self, checkProgress=False):
        print 'spawning nlp process using command:', self.start_corenlp        
            
        self.corenlp = pexpect.spawn(self.start_corenlp, maxread=8192,
                                     searchwindowsize=80)

        # show progress bar while loading the models
        # this is VERY sensitive to the models loaded and run mode, skip
        if checkProgress:
            widgets = ['Loading Models: ', Fraction()]
            pbar = ProgressBar(widgets=widgets, maxval=5,
                               force_update=True).start()
            # Model timeouts:
            # pos tagger model (~5sec)
            # NER-all classifier (~33sec)
            # NER-muc classifier (~60sec)
            # CoNLL classifier (~50sec)
            # PCFG (~3sec)
            timeouts = [20, 200, 600, 600, 20]
            for i in xrange(5):
                self.corenlp.expect("done.", timeout=timeouts[i])  # Load model
                pbar.update(i + 1)
            self.corenlp.expect(["Entering interactive shell.","\nNLP> "])
            pbar.finish()

        # check for interactive shell response
        self.corenlp.expect(["Processing will end when EOF is reached.",
                             pexpect.EOF, "\nNLP> "])

    def __init__(self, corenlp_path=DIRECTORY, memory="3g",
                 properties='default.properties', serving=False,
                 classname="edu.stanford.nlp.pipeline.StanfordCoreNLP"):
        """
        Checks the location of the jar files.
        Spawns the server as a process.
        """

        # spawn the server
        self.serving = serving
        self.start_corenlp = init_corenlp_command(corenlp_path, memory,
                                                  properties, 
                                                  classname=classname)
        self._spawn_corenlp()

    def close(self, force=True):
        self.corenlp.terminate(force)

    def isalive(self):
        return self.corenlp.isalive()

    def __del__(self):
        # If our child process is still around, kill it
        if self.isalive():
            self.close()

    def _parse(self, text):
        """
        This is the core interaction with the parser.

        It returns a Python data-structure, while the parse()
        function returns a JSON object
        """

        # CoreNLP interactive shell cannot recognize newline
        if '\n' in text or '\r' in text:
            to_send = re.sub("[\r\n]", " ", text).strip()
        else:
            to_send = text

        # clean up anything leftover
        def clean_up():
            while True:
                try:
                    self.corenlp.read_nonblocking(8192, 0.1)
                except pexpect.TIMEOUT:
                    break
                except pexpect.EOF:
                    break                
        clean_up()

        self.corenlp.sendline(to_send)

        # How much time should we give the parser to parse it?
        # the idea here is that you increase the timeout as a
        # function of the text's length.
        # max_expected_time = max(5.0, 3 + len(to_send) / 5.0)
        max_expected_time = max(300.0, len(to_send) / 3.0)

        # repeated_input = self.corenlp.except("\n")  # confirm it
        t = self.corenlp.expect(["\nNLP> ", "Negative", "Positive",
                                 pexpect.TIMEOUT, pexpect.EOF,
                                 "\nWARNING: Parsing of sentence failed, possibly because of out of memory."],
                                timeout=max_expected_time)
        incoming = self.corenlp.before

        if t == 3:
            # TIMEOUT, clean up anything left in buffer
            clean_up()
            print >>sys.stderr, {'error': "timed out after %f seconds" % max_expected_time,
                                 'input': to_send,
                                 'output': incoming}
            raise TimeoutError("Timed out after %d seconds" % max_expected_time)
        elif t == 4:
            # EOF, probably crash CoreNLP process
            print >>sys.stderr, {'error': "CoreNLP terminates abnormally while parsing",
                                 'input': to_send,
                                 'output': incoming}
            raise ProcessError("CoreNLP process terminates abnormally while parsing")
        elif t == 5:
            # out of memory
            print >>sys.stderr, {'error': "WARNING: Parsing of sentence failed, possibly because of out of memory.",
                                 'input': to_send,
                                 'output': incoming}
            raise OutOfMemoryError

        if VERBOSE:
            print "%s\n%s" % ('=' * 40, incoming)
        try:
            # parse output
            if t == 1:
                results = {"sentiment": "Negative"}
                results.update(parse_parser_sentimentLines(incoming))
            elif t == 2:
                results = {"sentiment": "Positive"}
                results.update(parse_parser_sentimentLines(incoming))
            else:   # default non-Sentiment mode results
                results = parse_parser_results(incoming)
        except Exception as e:
            if VERBOSE:
                print traceback.format_exc()
            raise e

        return results

    def raw_parse(self, text):
        """
        This function takes a text string, sends it to the Stanford parser,
        reads in the result, parses the results and returns a list
        with one dictionary entry for each parsed sentence.
        """
        try:
            r = self._parse(text)
            return r
        except Exception as e:
            print e  # Should probably log somewhere instead of printing
            self.corenlp.close()
            self._spawn_corenlp()
            if self.serving:  # We don't want to raise the exception when acting as a server
                return []
            raise e

    def parse(self, text):
        """
        This function takes a text string, sends it to the Stanford parser,
        reads in the result, parses the results and returns a list
        with one dictionary entry for each parsed sentence, in JSON format.
        """
        return json.dumps(self.raw_parse(text))


def batch_parse(input_folder, corenlp_path=DIRECTORY, properties='default.properties', raw_output=False, memory="3g"):
                
    """
    This function takes input files,
    sends list of input files to the Stanford parser,
    reads in the results from temporary folder in your OS and
    returns a generator object of list that consist of dictionary entry.
    If raw_output is true, the dictionary returned will correspond exactly to XML.
    ( The function needs xmltodict,
    and doesn't need init 'StanfordCoreNLP' class. )
    """
    if not os.path.exists(input_folder):
        raise Exception("input_folder does not exist")

    return parse_xml_output(input_folder, corenlp_path, memory,
                            raw_output=raw_output, properties=properties)


if __name__ == '__main__':
    """
    The code below starts an JSONRPC server
    """
    from jsonrpclib.SimpleJSONRPCServer import SimpleJSONRPCServer
    parser = optparse.OptionParser(usage="%prog [OPTIONS]")
    parser.add_option('-p', '--port', default='8080',
                      help='Port to serve on (default 8080)')
    parser.add_option('-H', '--host', default='127.0.0.1',
                      help='Host to serve on (default localhost; 0.0.0.0 to make public)')
    parser.add_option('-q', '--quiet', action='store_false', default=True,
                      dest='verbose',
                      help="Quiet mode, don't print status msgs to stdout")
    parser.add_option('-S', '--corenlp', default=DIRECTORY,
                      help='Stanford CoreNLP tool directory (default %s)' %\
                      DIRECTORY)
    parser.add_option('-s', '--sentiment', action='store_true', default=False,
                      dest="sentiment",
                      help='run sentiment analysis (only)')
    parser.add_option('-P', '--properties', default='default.properties',
                      help='Stanford CoreNLP properties fields (default: default.properties)')
    options, args = parser.parse_args()
    VERBOSE = options.verbose
    # server = jsonrpc.Server(jsonrpc.JsonRpc20(),
    #                         jsonrpc.TransportTcpIp(addr=(options.host, int(options.port))))

    classname = "edu.stanford.nlp.pipeline.StanfordCoreNLP"
    if (options.sentiment):
        classname = "edu.stanford.nlp.sentiment.SentimentPipeline"

    try:
        server = SimpleJSONRPCServer((options.host, int(options.port)))

        nlp = StanfordCoreNLP(options.corenlp, properties=options.properties,
                              serving=True, classname=classname)
        server.register_function(nlp.parse)
        server.register_function(nlp.raw_parse)

        print 'Serving on http://%s:%s' % (options.host, options.port)
        # server.serve()
        server.serve_forever()
    except KeyboardInterrupt:
        print >>sys.stderr, "Bye."
        exit()
