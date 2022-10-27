#Run tracking over the generated events with or without background
import argparse
import os
import pickle
import secrets
import subprocess
import sys


parser = argparse.ArgumentParser()
parser.add_argument('output', metavar='output_dir', type=str, help='file to save the resulting tracking parameters into')
parser.add_argument('input_files', metavar='events', nargs='+', type=str, help='file with events to perform tracking on')
parser.add_argument('--max-event', default=0, type=int, help='maximum number of events to process (0 for all)')

def run(output_file, input_file, max_event=0, seed=None):
    import basf2  # pylint: disable=import-error,wrong-import-position
    import reconstruction  # pylint: disable=import-error,wrong-import-position
    from ROOT import Belle2  # pylint: disable=import-error,wrong-import-position
    from tracking.validation.utilities import getHelixFromMCParticle  # pylint: disable=import-error,wrong-import-position


    def wrapped_getter(fn_getter):
        try:
            return fn_getter()
        except ReferenceError:
            return None

    class Factors(basf2.Module):
        KEYS = (
            'id_event',
            'id_track',
            'd0',
            'd0_t',
            'phi0',
            'phi0_t',
            'z0',
            'z0_t',
            'omega',
            'omega_t',
            'tlmd',
            'tlmd_t',
            'PXDHits',
            'SVDHits',
            'CDCHits',
            'pValue',
            'pt',
        )

        def __init__(self, save_path):
            super().__init__()
            self.save_path = save_path
            self.id_event = 0
            self.store = []

        def event(self):
            tracks = Belle2.PyStoreArray('Tracks')

            for id_track, track in enumerate(tracks):
                entry = {key: None for key in self.KEYS}
                track_rec = track.getRelated('RecoTracks')
                track_fit = track.getTrackFitResult(Belle2.Const.pion)

                if isinstance(track_fit, Belle2.TrackFitResult):
                    entry['d0'] = wrapped_getter(track_fit.getD0)
                    entry['omega'] = wrapped_getter(track_fit.getOmega)
                    entry['phi0'] = wrapped_getter(track_fit.getPhi0)
                    entry['tlmd'] = wrapped_getter(track_fit.getTanLambda)
                    entry['z0'] = wrapped_getter(track_fit.getZ0)
                    entry['pt'] = wrapped_getter(track_fit.getTransverseMomentum)
                    entry['pValue']= wrapped_getter(track_fit.getPValue)

                if isinstance(track_rec, Belle2.RecoTrack):
                    entry['PXDHits'] = track_rec.getNumberOfPXDHits()
                    entry['SVDHits'] = track_rec.getNumberOfSVDHits()
                    entry['CDCHits'] = track_rec.getNumberOfCDCHits()
                    mcparticle = track_rec.getRelated("MCParticles")

                    if isinstance(mcparticle, Belle2.MCParticle):
                        helix = getHelixFromMCParticle(mcparticle)
                        entry['d0_t'] = helix.getD0()
                        entry['omega_t'] = helix.getOmega()
                        entry['phi0_t'] = helix.getPhi0()
                        entry['tlmd_t'] = helix.getTanLambda()
                        entry['z0_t'] = helix.getZ0()


                entry['id_event'] = self.id_event
                entry['id_track'] = id_track

                for key, val in entry.items():
                    if val != val:
                        entry[key] = None

                self.store.append(entry)

            self.id_event += 1
            basf2.B2INFO(f'processed event #{self.id_event}')

        def terminate(self):
            with open(self.save_path, mode='wb') as file:
                pickle.dump(self.store, file)



    if seed is None:
        seed = secrets.randbelow(2**32 - 1)
    basf2.set_random_seed(seed)
    main = basf2.create_path()
    main.add_module('RootInput', inputFileName=input_file)
    main.add_module('EventInfoPrinter')
    main.add_module('Progress')
    main.add_module('Gearbox')
    main.add_module('Geometry')
    reconstruction.add_reconstruction(main)
    main.add_module(Factors(output_file))
    basf2.process(main, max_event)
    print(basf2.statistics)

if __name__ == '__main__':
    # parse args
    args = parser.parse_args()
    args.output = os.path.expandvars(args.output)

    # validate max event
    #The maximal number of events which will be processed, 0 for no limit
    if args.max_event < 0:
        parser.error(f'expecting max event > 0 (got: {args.max_event})')

    # validate input files
    args.input_files = [os.path.expandvars(path) for path in args.input_files]
    for input_file in args.input_files:
        if not os.path.exists(input_file):
            parser.error(f'no bkgoverlay file found at {input_file!r}')

    if len(args.input_files) > 1:
        # assume output is a directory
        if os.path.exists(args.output) and not os.path.isdir(args.output):
            parser.error(f'invalid output directory {args.output!r}')
        os.makedirs(args.output, exist_ok=True)

        # derive output files
        output_files = []
        for input_file in args.input_files:
            filename = os.path.splitext(os.path.basename(input_file))[0]
            output_file = os.path.join(args.output, f'{filename}.pkl')
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
        # infer output type
        input_file, output_file = next(iter(args.input_files)), args.output
        if os.path.exists(args.output):
            if os.path.isfile(args.output):
                # prevent overwriting
                parser.error(f'cannot overwrite existing file in {args.output!r}')
            elif os.path.isdir(args.output):
                filename = os.path.splitext(os.path.basename(input_file))[0]
                output_file = os.path.join(args.output, f'{filename}.pkl')
                if os.path.isfile(output_file):
                    # prevent overwriting
                    parser.error(f'cannot overwrite existing file in {output_file!r}')
            else:
                parser.error(f'invalid output path {args.output!r}')
        run(output_file, input_file, max_event=args.max_event)
