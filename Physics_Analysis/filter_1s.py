"""
Extraction of PXDDigits from background overlay
"""
import argparse
import os
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('output', metavar='output_dir', type=str)
parser.add_argument('input_files', metavar='bkgoverlay', nargs='+', type=str)
parser.add_argument('--max-event', default=0, type=int)

def run(output_file, input_file, max_event=0):
    import basf2  # pylint: disable=import-error,wrong-import-position
    main = basf2.create_path()
    main.add_module('RootInput', inputFileName=input_file, branchNames=['PXDDigits'])
    #Periodically writes the number of processed events/runs to the logging system.
    main.add_module('Progress')
    main.add_module('RootOutput', outputFileName=output_file, updateFileCatalog=False)
    basf2.process(main, max_event)

if __name__ == '__main__':
    # parse arguments
    args = parser.parse_args()
    args.output = os.path.expandvars(args.output)

    # validate max event
    if args.max_event < 0:
        parser.error(f'expecting max event > 0 (got: {args.max_event})')

    # validate input files
    args.input_files = [os.path.expandvars(path) for path in args.input_files]
    for input_file in args.input_files:
        if not os.path.exists(input_file):
            parser.error(f'no bkgoverlay file found at {input_file!r}')

    if len(args.input_files) > 1:
        if os.path.exists(args.output) and not os.path.isdir(args.output):
            parser.error(f'invalid output directory {args.output!r}')
        os.makedirs(args.output, exist_ok=True)

        output_files = []
        for input_file in args.input_files:
            filename = os.path.splitext(os.path.basename(input_file))[0]
            output_file = os.path.join(args.output, f'{filename}.root')
            # prevent overwriting
            if os.path.exists(output_file):
                parser.error(f'cannot overwrite existing file in {output_file!r}')
            output_files.append(output_file)

        proc = []
        for output_file, input_file in zip(output_files, args.input_files):
            command = ['python', sys.argv[0], output_file, input_file, '--max-event', str(args.max_event)]
            with open(output_file + '.log', 'w') as f:
                proc.append(subprocess.Popen(' '.join(command), encoding='utf-8', env=os.environ, shell=True, stderr=f, stdout=f))
        try:
            for p in proc:
                p.wait()
        except KeyboardInterrupt:
            for p in proc:
                p.kill()

    else:
        input_file, output_file = next(iter(args.input_files)), args.output
        if os.path.exists(args.output):
            if os.path.isfile(args.output):
                # prevent overwriting
                parser.error(f'cannot overwrite existing file in {args.output!r}')
            elif os.path.isdir(args.output):
                filename = os.path.splitext(os.path.basename(input_file))[0]
                output_file = os.path.join(args.output, f'{filename}.root')
                if os.path.isfile(output_file):
                    # prevent overwriting
                    parser.error(f'cannot overwrite existing file in {output_file!r}')
            else:
                parser.error(f'invalid path {args.output!r}')
        run(output_file, input_file, max_event=args.max_event)
