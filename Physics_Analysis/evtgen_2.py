import argparse
import json
import os
import secrets
import subprocess
import sys
import uuid

parser = argparse.ArgumentParser()
parser.add_argument('num_events', type=int)
parser.add_argument('output', metavar='output_dir', type=str)
parser.add_argument('input_files', metavar='bkgoverlay', nargs='*', type=str)
parser.add_argument('--num-jobs', default=1, type=int)

def run(output_file, num_events, *input_files, seed=None):
    import basf2  # pylint: disable=import-error
    import beamparameters  # pylint: disable=import-error
    import simulation  # pylint: disable=import-error
    import validation  # pylint: disable=import-error,unused-import

    if seed is None:
        seed = secrets.randbelow(2**32 - 1)
    basf2.set_random_seed(seed)

    main = basf2.create_path()
    main.add_module('EventInfoSetter', evtNumList=[num_events])
    main.add_module('EventInfoPrinter')
    beamparameters.add_beamparameters(main, 'Y4S')
    main.add_module('PrintBeamParameters')
    main.add_module('Progress')
    main.add_module('EvtGenInput')
    simulation.add_simulation(main, bkgfiles=list(input_files) if input_files else None)
    main.add_module('RootOutput', outputFileName=output_file, updateFileCatalog=False)
    basf2.process(main)

if __name__ == '__main__':
    # parse arguments
    args = parser.parse_args()
    args.output = os.path.expandvars(args.output)

    # validate num events
    if args.num_events < 0:
        parser.error(f'expecting num events > 0 (got: {args.num_events})')

    # validate num jobs
    if args.num_jobs < 0:
        parser.error(f'expecting num jobs  > 0 (got: {args.num_jobs})')

    # validate input files
    args.input_files = [os.path.expandvars(path) for path in args.input_files]
    for input_file in args.input_files:
        if not os.path.exists(input_file):
            parser.error(f'no bkgoverlay file found in {input_file!r}')

    if args.num_jobs > 1:
        if os.path.exists(args.output) and not os.path.isdir(args.output):
            parser.error(f'invalid output directory {args.output!r}')
        os.makedirs(args.output, exist_ok=True)

        proc = []
        for _ in range(args.num_jobs):
            filename = str(uuid.uuid4())
            output_file = os.path.join(args.output, f'{filename}.root')
            command = ['python', sys.argv[0], str(args.num_events), output_file, *args.input_files]
            with open(output_file + '.log', 'w') as f:
                proc.append(subprocess.Popen(' '.join(command), encoding='utf-8', env=os.environ, shell=True, stderr=f, stdout=f))
        try:
            for p in proc:
                p.wait()
        except KeyboardInterrupt:
            for p in proc:
                p.kill()
    else:
        output_file = args.output
        if os.path.exists(args.output):
            if os.path.isfile(args.output):
                # prevent overwriting
                parser.error(f'cannot overwrite existing file in {args.output!r}')
            elif os.path.isdir(args.output):
                filename = str(uuid.uuid4())
                output_file = os.path.join(args.output, f'{filename}.root')
            else:
                parser.error(f'invalid path {args.output!r}')

        with open(output_file + '.json', 'w') as f:
            json.dump(vars(args), f, indent=4, sort_keys=True)
        run(output_file, args.num_events, *args.input_files)
