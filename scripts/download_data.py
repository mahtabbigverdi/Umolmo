import argparse
import inspect
import logging
import time

from olmo.data.academic_datasets import ChartQa, ScienceQAImageOnly, TextVqa, OkVqa, DocQa, \
    InfoQa, AOkVqa, Vqa2, PlotQa, FigureQa, DvQa, SceneTextQa, TabWMPDirectAnswer, \
    AndroidControl, TallyQa, AI2D, CountBenchQa, RealWorldQa, MathVista, MMMU, ClockBench
from olmo.data.pixmo_datasets import PixMoDocs, PixMoCount, PixMoPoints, PixMoCapQa, PixMoCap, PixMoPointExplanations
from olmo.util import prepare_cli_environment

ACADEMIC_EVAL = [
    ChartQa, TextVqa, DocQa, InfoQa, Vqa2,
    AndroidControl, AI2D, CountBenchQa, RealWorldQa, MathVista, MMMU
]

ACADEMIC_DATASETS = [
    ChartQa, ScienceQAImageOnly, TextVqa, OkVqa, DocQa,
    InfoQa, AOkVqa, PlotQa, FigureQa, DvQa, SceneTextQa, TabWMPDirectAnswer,
    TallyQa, AI2D, CountBenchQa, RealWorldQa, MathVista, MMMU,
    Vqa2, AndroidControl
]

PIXMO_DATASETS = [
    PixMoDocs, PixMoCount, PixMoPoints, PixMoCapQa, PixMoCap, PixMoPointExplanations
]

DATASETS = ACADEMIC_DATASETS + PIXMO_DATASETS


DATASET_MAP = {
    x.__name__.lower(): x for x in DATASETS
}


def download():
    parser = argparse.ArgumentParser(prog="Download Molmo datasets")
    parser.add_argument("dataset",
                        help="Datasets to download, can be a name or one of: all, pixmo, academic or academic_eval")
    parser.add_argument("--n_procs", type=int, default=1,
                        help="Number of processes to download with")
    args = parser.parse_args()

    prepare_cli_environment()

    if args.dataset == "all":
        to_download = DATASETS
    elif args.dataset == "academic":
        to_download = ACADEMIC_DATASETS
    elif args.dataset == "pixmo":
        to_download = PIXMO_DATASETS
    elif args.dataset == "academic_eval":
        to_download = ACADEMIC_EVAL
    elif args.dataset.lower().replace("_", "") in DATASET_MAP:
        to_download = [DATASET_MAP[args.dataset.lower().replace("_", "")]]
    else:
        raise NotImplementedError(args.dataset)

    for ix, dataset in enumerate(to_download):
        t0 = time.perf_counter()
        logging.info(f"Starting download for {dataset.__name__} ({ix+1}/{len(to_download)})")
        dataset.download(n_procs=args.n_procs)
        logging.info(f"Done with {dataset.__name__} in {time.perf_counter()-t0:0.1f} seconds")


if __name__ == '__main__':
    download()
