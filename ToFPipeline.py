import yaml
import numpy as np
import xarray as xr
import pandas as pd
from scipy.signal import find_peaks
from functools import partial

from pathlib import Path
import os, re
import h5py
import dask.array as da

from tqdm.notebook import tqdm

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt



class GlobalConfig:
    _config = {}

    @classmethod
    def load(cls, path):
        """Load a YAML config file into memory."""
        with open(path, "r") as f:
            cls._config = yaml.safe_load(f) or {}

    @classmethod
    def get_for_class(cls, cls_or_name):
        """Return the config dictionary for a given class."""
        name = cls_or_name if isinstance(cls_or_name, str) else cls_or_name.__name__
        return cls._config.get(name, {}).copy()

class Configurable:
    CONFIG_KEY = None
    def __init__(self, config=None):
        self.className = self.__class__.__name__
        class_key = self.CONFIG_KEY or self.__class__.__name__
        base = GlobalConfig.get_for_class(class_key)
        if config:
            base.update(config)
        self.config = base

    def getConfig(self, key, default=None):
        return self.config.get(key, default)

class Loader(Configurable):
    def load(self):
        raise NotImplementedError

    def sampleShift(self, shift=None, mode=None):
        """
        Shift each detector's trace along the sample axis.

        Positive shift → data moves right (toward higher sample indices).
        Negative shift → data moves left  (toward lower sample indices).

        Parameters
        ----------
        shift : int, or dict {detector: int}, optional
            Scalar – same shift applied to every detector.
            Dict   – per-detector shifts, e.g. ``{0: 5, 1: -3}``.
            Detectors not listed in a dict receive no shift.
            Falls back to the ``sampleShift`` key in the config.
            ``0`` or an empty dict means no shift is applied.
        mode : {'zero', 'roll'}, optional
            ``'zero'``  – vacated positions are filled with zeros (default).
            ``'roll'``  – data wraps around the trace (circular shift).
            Falls back to the ``sampleShiftMode`` key in the config.

        Returns
        -------
        self
        """
        shift = shift if shift is not None else self.config.get("sampleShift", 0)
        mode  = mode  if mode  is not None else self.config.get("sampleShiftMode", "zero")

        if shift == 0 or shift == {} or shift is None:
            return self

        sampleCoords = self.data.coords["sample"]
        roll = (mode == "roll")

        def _applyShift(arr, n):
            """Shift a dask array along its last axis by n samples."""
            if n == 0:
                return arr
            if roll:
                return da.roll(arr, n, axis=-1)
            else:
                nSamples = arr.shape[-1]
                pad_width = [(0, 0)] * (arr.ndim - 1)
                if n > 0:
                    pad_width.append((n, 0))
                    padded = da.pad(arr, pad_width, mode="constant", constant_values=0)
                    return padded[..., :nSamples]
                else:
                    pad_width.append((0, -n))
                    padded = da.pad(arr, pad_width, mode="constant", constant_values=0)
                    return padded[..., -n:]

        if isinstance(shift, dict):
            shiftMap = {int(k): int(v) for k, v in shift.items()}
            arrays = []
            for det in self.data.detector.values:
                detData = self.data.sel(detector=det)
                detShift = shiftMap.get(int(det), 0)
                if detShift != 0:
                    shifted = _applyShift(detData.data, detShift)
                    detData = detData.copy(data=shifted)
                arrays.append(detData)
            self.data = xr.concat(arrays, dim="detector")
        else:
            shifted = _applyShift(self.data.data, int(shift))
            self.data = self.data.copy(data=shifted)

        # Sample coordinate labels are fixed; only the values shift
        self.data = self.data.assign_coords(sample=sampleCoords)
        return self

    def pulseShift(self, shift=None, mode=None):
        """
        Shift each pulse's trace along the sample axis.

        Positive shift → data moves right (toward higher sample indices).
        Negative shift → data moves left  (toward lower sample indices).

        Parameters
        ----------
        shift : int, or dict {pulseId: int}, optional
            Scalar – same shift applied to every pulse.
            Dict   – per-pulse shifts keyed by pulseId, e.g. ``{0: 5, 1: -3}``.
            Pulses not listed in a dict receive no shift.
            Falls back to the ``pulseShift`` key in the config.
            ``0`` or an empty dict means no shift is applied.
        mode : {'zero', 'roll'}, optional
            ``'zero'``  – vacated positions are filled with zeros (default).
            ``'roll'``  – data wraps around the trace (circular shift).
            Falls back to the ``pulseShiftMode`` key in the config.

        Returns
        -------
        self
        """
        shift = shift if shift is not None else self.config.get("pulseShift", 0)
        mode  = mode  if mode  is not None else self.config.get("pulseShiftMode", "zero")

        if shift == 0 or shift == {} or shift is None:
            return self

        sampleCoords = self.data.coords["sample"]
        roll = (mode == "roll")

        def _applyShift(arr, n):
            if n == 0:
                return arr
            if roll:
                return da.roll(arr, n, axis=-1)
            else:
                nSamples = arr.shape[-1]
                pad_width = [(0, 0)] * (arr.ndim - 1)
                if n > 0:
                    pad_width.append((n, 0))
                    padded = da.pad(arr, pad_width, mode="constant", constant_values=0)
                    return padded[..., :nSamples]
                else:
                    pad_width.append((0, -n))
                    padded = da.pad(arr, pad_width, mode="constant", constant_values=0)
                    return padded[..., -n:]

        if isinstance(shift, dict):
            shiftMap = {int(k): int(v) for k, v in shift.items()}
            pulseIndex = self.data.indexes["pulse"]
            pulseIds = pulseIndex.get_level_values("pulseId")
            arrays = []
            for i, pid in enumerate(pulseIds):
                pulseData = self.data.isel(pulse=i)
                pulseShiftVal = shiftMap.get(int(pid), 0)
                if pulseShiftVal != 0:
                    shifted = _applyShift(pulseData.data, pulseShiftVal)
                    pulseData = pulseData.copy(data=shifted)
                arrays.append(pulseData)
            self.data = xr.concat(arrays, dim="pulse")
            # Restore the original pulse MultiIndex (xr.concat replaces it with integers)
            mindex_coords = xr.Coordinates.from_pandas_multiindex(pulseIndex, "pulse")
            self.data = self.data.assign_coords(mindex_coords)
        else:
            shifted = _applyShift(self.data.data, int(shift))
            self.data = self.data.copy(data=shifted)

        self.data = self.data.assign_coords(sample=sampleCoords)
        return self

_RUN_CACHE = {}
class FLASHLoader(Loader):
    def __init__(self,proposal, runNo,config=None):
        super().__init__(config)
        if self.className == "FLASHLoader":
            #print("Using ",self.className)
            import dask.array as da
            from fab.magic import config, beamtime, ballchamber, opis, timing
            self.da = da
            self.ballchamber = ballchamber
            self.opis = opis
        self.proposal = proposal
        self.runNo = runNo
        self._run = None
        self.key = None
        self.data = None
        fullTheta = np.array(self.config.get("angles", np.linspace(0,2*np.pi,16,endpoint=False)))
        detectors = self.config.get("ToF",range(16))
        theta = fullTheta[detectors]
        self.angles = pd.DataFrame(detectors,columns=["detector"])
        self.angles["Angles"] = theta
        self.xmg = None
        self.photonEnergy = None
    
    def load(self, key="all", trainStart=None, trainStop=None, trainStep = None, pulseStart=None, pulseStop=None, pulseStep=None,roi=[None,None]):
        self.key = key or self.config.get("key", "all")
        trainStart = trainStart or self.config.get("trainStart", None)
        trainStop = trainStop or self.config.get("trainStop", None)
        trainStep = trainStep or self.config.get("trainStep", None)
        pulseStart = pulseStart or self.config.get("pulseStart", None)
        pulseStop = pulseStop or self.config.get("pulseStop", None)
        pulseStep = pulseStep or self.config.get("pulseStep", None)
        
        traces = self.ballchamber.load(daq_run=self.runNo)
        
        if any(v is not None for v in [trainStart, trainStop, trainStep]):
            traces = traces.isel(train_id=slice(trainStart, trainStop, trainStep))
        if any(v is not None for v in [pulseStart, pulseStop, pulseStep]):
            traces = traces.isel(shot_id=slice(pulseStart, pulseStop, pulseStep))
        
        # ds is your original dataset
        
        # 1) Stack train_id + shot_id into a single pulse MultiIndex
        ds2 = traces.stack(pulse=("train_id", "shot_id"))
        
        # Optional: rename MultiIndex levels
        ds2 = ds2.rename({"train_id": "trainId", "shot_id": "pulseId"})
        ds2 = ds2.set_index(pulse=("trainId", "pulseId"))
        
        # 2) Rename tof_trace → sample
        ds2 = ds2.rename({"tof_trace": "sample"})
        
        # 3) Combine adcXX variables into a new detector dimension
        detector_names = [f"adc{str(i).zfill(2)}" for i in range(16)]
        
        ds_stacked = xr.concat([ds2[v] for v in detector_names], dim="detector")
        ds_stacked = ds_stacked.assign_coords(detector=np.arange(16))
        
        # 4) Final reorder
        self.run = ds_stacked.transpose("detector", "pulse", "sample")
        
        # Assign integer sample coordinates to preserve original indices when slicing
        nSamples = self.run.sizes["sample"]
        self.run = self.run.assign_coords(sample=np.arange(nSamples, dtype=np.int64))
        
        self.data = self.run.sel(sample=slice(roi[0],roi[1]))

        if self.key in ("all", "Photon Energy"):
            self.photonEnergy = self.opis.load(daq_run=self.runNo).to_dataframe().reset_index()
            self.photonEnergy.columns = ["trainId","daq_run","Photon Energy"]
            fullTrainIds = pd.RangeIndex(self.photonEnergy["trainId"].min(), self.photonEnergy["trainId"].max()+1)
            self.photonEnergy = self.photonEnergy.set_index("trainId").reindex(fullTrainIds)
            self.photonEnergy["Photon Energy"] = self.photonEnergy["Photon Energy"].interpolate()
            self.photonEnergy = (self.photonEnergy.reset_index().rename(columns={"index": "trainId"}))

        return self

    def defaultPreprocessing(self,ToF=None,baselineRegion=None,trainStart=None,trainStop=None):
        ToF = ToF or self.config.get("ToF", [0])
        self.data = self.data.sel(detector=ToF)
        baselineRegion = baselineRegion or self.config.get("baselineRegion",[-200,None])
        self.data = -(self.data - self.data.isel(sample=slice(baselineRegion[0],baselineRegion[1])).mean(dim="sample"))
        grouped = self.data.groupby("daq_run")
        sliced = xr.concat(
            [g.isel(pulse=slice(trainStart, trainStop)) for _, g in grouped],
            dim="pulse")
        self.data = sliced
        return self      
class EuXFelLoader(Loader):
    def __init__(self,proposal, runNo,config=None):
        super().__init__(config)
        print("init...")
        if self.className == "EuXFelLoader":
            print("Using ",self.className)
            import extra_data as xd
            from euxfel_bunch_pattern import indices_at_sase
            from extra.components import AdqRawChannel
            self.xd = xd
            self.indices_at_sase = indices_at_sase
            self.AdqRawChannel = AdqRawChannel
        self.proposal = proposal
        self.runNo = runNo
        self._run = None
        self.key = None
        self.data = None
        self.xmg = None
        self.photonEnergy = None
        print("Done!")

    @property
    def run(self):
        global _RUN_CACHE
        cacheKey = (self.proposal, self.runNo)
        if self._run is not None:
            return self._run
        if cacheKey in _RUN_CACHE:
            self._run = _RUN_CACHE[cacheKey]
            return self._run

        print("opening run: ",self.runNo)
        self._run = self.xd.open_run(proposal=self.proposal, run=self.runNo)
        _RUN_CACHE[cacheKey] = self._run
        return self._run

    def filterByIntensity(self,intensityThreshold=None,):
        intensityThreshold = (
            intensityThreshold
            or self.config.get("intensityThreshold", None))
        if intensityThreshold is not None:
            mask = self.xgm > intensityThreshold
            self.data = self.data.where(mask, drop=True)                 
        return self

    
    def load(self, key="all", trainStart=None, trainStop=None, trainStep=None):
        self.key = key or self.config.get("key", "all")
        trainStart = trainStart or self.config.get("trainStart", 0)
        trainStop = trainStop or self.config.get("trainStop", -1)
        trainStep = trainStep or self.config.get("trainStep", 1)
        self.data = self.run[trainStart:trainStop:trainStep]
        if self.key in ("all", "XGM"):
            self.xgm = self.run[trainStart:trainStop:trainStep].select('SA3_XTD10_XGM/XGM/DOOCS:output')['SA3_XTD10_XGM/XGM/DOOCS:output', 'data.intensitySa3TD'].xarray()
        if self.key  in ("all", "Photon Energy"):
            self.photonEnergy = self.data.select("SA3_XTD10_UND/DOOCS/PHOTON_ENERGY_COLOR2","calibratedActualPosition").get_dataframe().reset_index()
            self.photonEnergy.columns = ["trainId","Photon Energy"]
        else:
            raise KeyError(f"{self.key} not found in loader")
        return self

    def detectors(self, proposal, run):
        """
        Provides detector information when given a run number.
        
        Parameters
        ----------
        run : unsigned int
            Run number within proposal
        
        Returns
        -------
        detinfo : structured ndarray
            Keys: name (detector name), 
                  digitizer,
                  channel,
                  angle (degrees, looking along the beam, 0 is right, increasing counter-clockwise),
                  sample_rate (GS/s)
        """
        confs = os.listdir(Path(__file__).parent / 'configurations' / str(proposal))
        groups = [re.search('(\d*)-(\d*).txt', f) for f in confs]
        for idx, gr in enumerate(groups):
            if gr is not None:
                a, b = gr.group(1, 2)
                if (run >= int(a)) & (run <= int(b)):
                    #print(f"Using configuration file: {Path(__file__).parent / 'configurations' / str(proposal) / confs[idx]}")
                    return  np.genfromtxt(Path(__file__).parent / 'configurations' / str(proposal) / confs[idx],
                                          names=True, dtype=('|U5', '|U4', '|U3', '<f8', '?', '<i4'))
        raise Exception(f'Did not find detector configuration file for run {run}')

    def offsets(self, proposal, run):
        from_, to, train_offset, pulse_length = np.genfromtxt(Path(__file__).parent / 'configurations' / str(proposal) / 'offsets.cfg', unpack=True)
        idx = np.argmax((run >= from_) & (run <= to))
        return train_offset[idx], int(pulse_length[idx])

    def defaultPreprocessing(self,ToF=None,
        pattern_noise_region=None,
        pattern_noise_sym=None):

        ToF = (
            ToF
            if ToF is not None
            else self.config.get("ToF", [0])
        )

        pattern_noise_region = (
            pattern_noise_region
            if pattern_noise_region is not None
            else self.config.get("pattern_noise_region", np.s_[:1000])
        )
        pattern_noise_sym = (
            pattern_noise_sym
            if pattern_noise_sym is not None
            else self.config.get("pattern_noise_sym", 8)
        )

        
        det = self.detectors(self.proposal, self.runNo)
        det = det[ToF]
        offs = self.offsets(self.proposal, self.runNo)

        self.data = xr.concat([
            self.AdqRawChannel(
                self.data,
                d['channel'],
                digitizer=f'SQS_DIGITIZER_{d["digitizer"]}',
                first_pulse_offset=d["first_pulse_offset"],
                single_pulse_length=offs[1],
                baseline=pattern_noise_region,
                cm_period=pattern_noise_sym
            ).pulse_data()[..., :offs[1]]
            for d in tqdm(det,position=1,leave=False,disable=True)
        ], dim=pd.Index(ToF,name="detector"))
        self.data = -self.data
        """
        if self.key == "all" or "XGM":
            lenTrain = 380
            pulseIds = self.data.pulseId.values.reshape(-1,lenTrain)[0]
            self.xgm = self.xgm.rename({"dim_0": "pulseId"}).isel(pulseId=slice(0,lenTrain))
            self.xgm = self.xgm.assign_coords(pulseId = pulseIds).stack(pulse=('trainId','pulseId'))
        """
        self.sampleShift()
        return self
class NXSLoader(Loader):
    """
    Loader for .nxs files containing time-of-flight histogram data.
    
    Each .nxs file represents a single run with histogram data from 15 channels.
    The data is transformed into an xarray.DataArray with dimensions:
    - detector: Channel numbers (0-14)
    - pulse: Measurement points per run  
    - sample: Time-of-flight bins
    """
    
    def __init__(self, dataPath, runNumbers=None, config=None):
        super().__init__(config)
        self.dataPath = Path(dataPath)
        self.runNumbers = runNumbers or []
        self.data = None
        self._filePattern = self.config.get("filePattern", "*_{run_number:05d}.nxs")
        
        # Set up detector angles if provided in config
        fullTheta = np.array(self.config.get("angles", np.linspace(0, 360, 16, endpoint=False)))
        detectors = self.config.get("ToF",[1])
        if len(fullTheta) == len(detectors):
            # Angles already correspond 1:1 to the selected detectors
            theta = fullTheta
        else:
            # Angles is a full set — index by detector number
            theta = fullTheta[detectors]
        self.angles = pd.DataFrame(detectors, columns=["detector"])
        self.angles["Angles"] = theta
    
    def _getFilePath(self, runNumber):
        """Get file path for a given run number."""
        pattern = self._filePattern.format(run_number=runNumber)
        files = list(self.dataPath.glob(pattern))
        if not files:
            # Try alternative patterns
            altPatterns = [
                f"*{runNumber:05d}.nxs",
                f"*_{runNumber:05d}.nxs", 
                f"*{runNumber}.nxs"
            ]
            for altPattern in altPatterns:
                files = list(self.dataPath.glob(altPattern))
                if files:
                    break
        if not files:
            raise FileNotFoundError(f"No .nxs file found for run {runNumber}")
        return files[0]
    
    def _extractRunNumber(self, filePath):
        """Extract run number from filename."""
        match = re.search(r'_(\d+)\.nxs$', str(filePath))
        return int(match.group(1)) if match else None
    
    def _loadSingleFile(self, filePath, roi=None):
        """Load data from a single .nxs file."""
        with h5py.File(filePath, 'r') as f:
            runNumber = self._extractRunNumber(filePath)
            if runNumber is None:
                raise ValueError(f"Could not extract run number from {filePath}")
            
            # Get data from all histogram channels
            channelsData = []
            nDetectors = 15  # ch01 to ch15
            
            for chIdx in range(1, nDetectors + 1):
                chName = f'histogram_ch{chIdx:02d}'
                if f'scan/instrument/{chName}/data' in f:
                    # Shape is (n_pulses, n_samples)
                    chData = f[f'scan/instrument/{chName}/data'][:]
                    # Apply ROI if specified
                    if roi is not None:
                        chData = chData[:, roi[0]:roi[1]]
                    channelsData.append(chData)
                else:
                    # Handle missing channels by creating zeros
                    print(f"Warning: {chName} not found in {filePath}")
                    if channelsData:
                        chData = np.zeros_like(channelsData[0])
                    else:
                        # Apply ROI to default shape if specified
                        if roi is not None:
                            defaultShape = (2, roi[1] - roi[0])
                        else:
                            defaultShape = (2, 1920)  # Default shape based on observed data
                        chData = np.zeros(defaultShape)
                    channelsData.append(chData)
            
            # Stack channel data: (n_detectors, n_pulses, n_samples)  
            dataArray = np.stack(channelsData, axis=0)
            
            # Get time-of-flight axis - use first available channel
            for chIdx in range(1, nDetectors + 1):
                chName = f'histogram_ch{chIdx:02d}'
                if f'scan/instrument/{chName}/time_of_flight' in f:
                    tofAxis = f[f'scan/instrument/{chName}/time_of_flight'][:]
                    # Apply ROI to tof axis if specified
                    if roi is not None:
                        tofAxis = tofAxis[roi[0]:roi[1]]
                    break
            else:
                # Fallback: create default ToF axis
                nSamples = dataArray.shape[2]
                tofAxis = np.linspace(0, nSamples * 0.1, nSamples)
                # If ROI was applied, adjust the start of the axis
                if roi is not None:
                    tofAxis = tofAxis + (roi[0] * 0.1)
            
            # Get timestamps
            timestamps = f['scan/instrument/collection/timestamp'][:]
            nPulses = dataArray.shape[1]
            
            return {
                'data': dataArray,
                'runNumber': runNumber, 
                'tofAxis': tofAxis,
                'timestamps': timestamps,
                'nPulses': nPulses,
                'roi': roi  # Store ROI for coordinate creation
            }
    
    def load(self, runNumbers=None, roi=None):
        """
        Load data from .nxs files.
        
        Parameters
        ----------
        runNumbers : list, optional
            List of run numbers to load. If None, uses self.runNumbers.
        roi : tuple or list, optional
            Region of interest for sample axis as [start, end]. If None, loads all samples.
            Falls back to ``roi`` key in config (e.g. ``roi: [0, 1000]``).
            Original sample indices are preserved in coordinates.
            
        Returns
        -------
        self : NXSLoader
            Returns self with loaded data in self.data
        """
        roi = roi if roi is not None else self.config.get('roi', None)
        runNumbers = runNumbers or self.runNumbers
        if not runNumbers:
            # Auto-detect run numbers from files
            nxsFiles = list(self.dataPath.glob("*.nxs"))
            runNumbers = []
            for filePath in nxsFiles:
                runNum = self._extractRunNumber(filePath)
                if runNum is not None:
                    runNumbers.append(runNum)
            runNumbers = sorted(runNumbers)
            print(f"Auto-detected {len(runNumbers)} runs: {runNumbers[:5]}{'...' if len(runNumbers) > 5 else ''}")
        
        allData = []
        allCoords = {
            'daq_run': [],
            'trainId': [],
            'pulseId': []
        }
        
        for runNum in runNumbers:
            try:
                filePath = self._getFilePath(runNum) 
                fileData = self._loadSingleFile(filePath, roi=roi)
                
                data = fileData['data']
                nPulses = fileData['nPulses']
                
                # Create pulse coordinates for this run
                # Each pulse gets a unique trainId and pulseId 
                trainIds = np.full(nPulses, runNum, dtype=np.uint32)
                pulseIds = np.arange(nPulses, dtype=np.int64)
                daqRuns = np.full(nPulses, runNum, dtype=np.uint32)
                
                allData.append(data)
                allCoords['daq_run'].extend(daqRuns)
                allCoords['trainId'].extend(trainIds)
                allCoords['pulseId'].extend(pulseIds)
                
            except Exception as e:
                print(f"Error loading run {runNum}: {e}")
                continue
        
        if not allData:
            raise ValueError("No data could be loaded")
        
        # Concatenate all data along pulse dimension
        fullData = np.concatenate(allData, axis=1)
        
        # Create dask array with appropriate chunking
        chunkSize = (1, min(len(allCoords['trainId']), 21175), fullData.shape[2])
        daskData = da.from_array(fullData, chunks=chunkSize)
        
        # Get sample axis from first loaded file
        filePath = self._getFilePath(runNumbers[0])
        sampleData = self._loadSingleFile(filePath, roi=roi)
        nSamples = len(sampleData['tofAxis'])
        
        # Create coordinates - maintain original sample indices if ROI was used
        detectorCoords = np.arange(fullData.shape[0], dtype=np.int64)
        if roi is not None:
            # Preserve original sample indices
            sampleCoords = np.arange(roi[0], roi[1], dtype=np.int64)
        else:
            sampleCoords = np.arange(nSamples, dtype=np.int64)
        
        # Create MultiIndex for pulse coordinate
        pulseTuples = list(zip(allCoords['trainId'], allCoords['pulseId']))
        pulseIndex = pd.MultiIndex.from_tuples(
            pulseTuples, 
            names=['trainId', 'pulseId']
        )
        
        # Create xarray DataArray
        self.data = xr.DataArray(
            daskData,
            dims=['detector', 'pulse', 'sample'],
            coords={
                'detector': ('detector', detectorCoords),
                'pulse': ('pulse', pulseIndex),
                'sample': ('sample', sampleCoords),
                'daq_run': ('pulse', allCoords['daq_run'])
            },
            name='adc00'
        )
        
        return self
    
    def defaultPreprocessing(self, ToF=None, baselineRegion=None, trainStart=None, trainStop=None):
        """
        Apply default preprocessing similar to other loaders.
        """
        ToF = ToF or self.config.get("ToF", [0])
        self.data = self.data.sel(detector=ToF)
        
        if baselineRegion is not None:
            baselineRegion = baselineRegion
        else:
            baselineRegion = self.config.get("baselineRegion", [0, 10])
        
        # Apply baseline correction
        if baselineRegion[1] is None:
            baselineSlice = slice(baselineRegion[0], None)
        else:
            baselineSlice = slice(baselineRegion[0], baselineRegion[1])
            
        self.data = (self.data - self.data.isel(sample=baselineSlice).mean(dim="sample"))
        
        # Apply train slicing if specified
        if trainStart is not None or trainStop is not None:
            grouped = self.data.groupby("daq_run")
            sliced = xr.concat(
                [g.isel(pulse=slice(trainStart, trainStop)) for _, g in grouped],
                dim="pulse"
            )
            self.data = sliced

        self.sampleShift()
        return self
class PeakFinder(Configurable):
    def __init__(self, data, config=None):
        super().__init__(config)
        self.data = data
        self.results = None

    def stack(self, stackTrains=None, trainStackSize=None, stackPulses=None, pulseStackStart=None, pulseStackStop=None, pulseStackSize=None, pulseStackStep=None):
        pulseIndex = self.data["pulse"].to_index()
        trainIds = pulseIndex.get_level_values("trainId").to_numpy()
        pulseIds = pulseIndex.get_level_values("pulseId").to_numpy()
        
        stackTrains = stackTrains if stackTrains is not None else self.config.get("stackTrains", True)
        trainStackSize = trainStackSize if trainStackSize is not None else self.config.get("trainStackSize", len(np.unique(trainIds)))

        stackPulses = stackPulses if stackPulses is not None else self.config.get("stackPulses", True)
        pulseStackStart = pulseStackStart if pulseStackStart is not None else self.config.get("pulseStackStart", 0)
        pulseStackStop = pulseStackStop if pulseStackStop is not None else self.config.get("pulseStackStop", len(np.unique(pulseIds)))
        # pulseStackStep is an alias for pulseStackSize used in stream/GUI configs
        if pulseStackSize is None:
            pulseStackSize = pulseStackStep if pulseStackStep is not None else None
        if pulseStackSize is None:
            pulseStackSize = self.config.get("pulseStackSize", None)
        if pulseStackSize is None:
            pulseStackSize = self.config.get("pulseStackStep", len(np.unique(pulseIds)))
        
        if trainStackSize is None:
            trainStackSize = len(np.unique(trainIds))
        if pulseStackStart is None:
            pulseStackStart = 0
        if pulseStackSize is None:
            pulseStackSize = len(np.unique(pulseIds))
        if pulseStackStop is None:
            pulseStackStop = len(np.unique(pulseIds))        
        
        if stackTrains == True:
            chunks = []
            uniqueTrainIds = np.unique(trainIds)
                
            nTrainStacks = len(uniqueTrainIds)//trainStackSize
            if nTrainStacks == 0:
                raise ValueError("pulseStackSize larger than available trains")
            
            for i in range(nTrainStacks):
                selTrainIds = uniqueTrainIds[i*trainStackSize:(i+1)*trainStackSize]
                trainMask = np.isin(self.data.trainId, selTrainIds)
                chunk = self.data.isel(pulse=trainMask).groupby("pulseId").mean()
                chunkTrainId = [int(selTrainIds[0])]
                chunk = chunk.expand_dims("trainId")
                chunk = chunk.assign_coords(trainId=("trainId",chunkTrainId))
                
                chunkTrainIds = selTrainIds[0]
                chunkPulseIds = chunk.pulseId.values
                chunkTrainIds = [int(chunkTrainIds)]*len(chunkPulseIds)
                chunk = chunk.rename({"trainId": "tid", "pulseId": "pulse"})
                multi_idx = pd.MultiIndex.from_arrays([chunkTrainIds, chunkPulseIds],names=["trainId", "pulseId"])
                mindex_coords = xr.Coordinates.from_pandas_multiindex(multi_idx, 'pulse')
                chunk = chunk.assign_coords(mindex_coords).drop_vars("tid").squeeze("tid") 
                chunks.append(chunk)
                
            stack = xr.concat(chunks,dim="pulse")
            self.data = stack
    
        if stackPulses:
            chunks = []
            uniquePulsIds = np.unique(pulseIds)
            nPulseStacks = len(uniquePulsIds)//pulseStackSize
            if nPulseStacks == 0:
                raise ValueError("pulseStackSize larger than available trains")
            
            for i in range(pulseStackStart,pulseStackStop,pulseStackSize):
                selPulseIds = uniquePulsIds[i:i+1]
                pulseMask = np.isin(self.data.pulseId, selPulseIds)
                chunk = self.data.isel(pulse=pulseMask).groupby("trainId").mean()
                chunkPulseId = [int(selPulseIds[0])]
                chunk = chunk.expand_dims("pulseId")
                chunk = chunk.assign_coords(pulseId=("pulseId",chunkPulseId))
                
                chunkPulseIds = selPulseIds[0]
                chunkTrainIds = chunk.trainId.values
                chunkPulseIds = [int(chunkPulseIds)]*len(chunkTrainIds)
                chunk = chunk.rename({"trainId": "pulse", "pulseId": "pid"})
                multi_idx = pd.MultiIndex.from_arrays([chunkTrainIds, chunkPulseIds],names=["trainId", "pulseId"])
                mindex_coords = xr.Coordinates.from_pandas_multiindex(multi_idx, 'pulse')
                chunk = chunk.assign_coords(mindex_coords).drop_vars("pid").squeeze("pid")  
                chunks.append(chunk)
    
            stack = xr.concat(chunks,dim="pulse")
            self.data = stack
        return self
        

    def normalize(self,ToF=None):
        if ToF != None:
            self.data = self.data / self.data.sel(detector=ToF).max().compute()
        else:
            self.data = self.data/self.data.max().compute()
        self.data = self.data.persist()
        return self

    def smooth(self, windowSize=None):
        """
        Apply smoothing to the data using a rolling average.
        
        Parameters
        ----------
        windowSize : int, optional
            Size of the rolling window for smoothing. If None, uses config value or defaults to 5.
            
        Returns
        -------
        self : PeakFinder
            Returns self for method chaining
        """
        windowSize = windowSize if windowSize is not None else self.config.get("smoothWindow", 5)
        
        # Apply rolling mean along the sample dimension, only where we have enough points
        smoothed = self.data.rolling(sample=windowSize, center=True, min_periods=windowSize).mean()
        
        # Fill NaN values (at edges) with original data to preserve length
        self.data = smoothed.fillna(self.data)
        
        # Rechunk sample dimension to single chunk for downstream processing
        self.data = self.data.chunk({"sample": -1})
        
        self.data = self.data.persist()
        return self

    def tShift(self, shift=None):
        """
        Shift each detector along the sample axis with zero-padding.

        Positive shift → data moves right; zeros fill the beginning, tail is cut.
        Negative shift → data moves left;  zeros fill the end,       head is cut.

        Parameters
        ----------
        shift : int, or dict {detector: int}, optional
            Scalar – same shift applied to every detector.
            Dict   – per-detector shifts, e.g. {0: 5, 1: -3}.
            Detectors not listed in a dict are not shifted.
            Falls back to the ``tShift`` key in the config (same types accepted).
        """
        def _shiftArray(arr, n):
            """Shift a dask/numpy array along its last axis by n, zero-padding."""
            if n == 0:
                return arr
            nSamples = arr.shape[-1]
            pad_width = [(0, 0)] * (arr.ndim - 1)
            if n > 0:
                pad_width.append((n, 0))
                padded = da.pad(arr, pad_width, mode="constant", constant_values=0)
                return padded[..., :nSamples]
            else:
                pad_width.append((0, -n))
                padded = da.pad(arr, pad_width, mode="constant", constant_values=0)
                return padded[..., -n:]

        shift = shift if shift is not None else self.config.get("tShift", 0)
        sampleCoords = self.data.coords["sample"]

        if isinstance(shift, dict):
            shiftMap = {int(k): int(v) for k, v in shift.items()}
            arrays = []
            for det in self.data.detector.values:
                detData = self.data.sel(detector=det)
                detShift = shiftMap.get(int(det), 0)
                if detShift != 0:
                    shifted = _shiftArray(detData.data, detShift)
                    detData = detData.copy(data=shifted)
                arrays.append(detData)
            self.data = xr.concat(arrays, dim="detector")
        else:
            n = int(shift)
            shifted = _shiftArray(self.data.data, n)
            self.data = self.data.copy(data=shifted)

        # Restore original sample coordinates (they don't change, only values shift)
        self.data = self.data.assign_coords(sample=sampleCoords)
        self.data = self.data.persist()
        return self
    
    def process(self, threshold=None, peakNo=None,roi=None, distanceFactor=None,widthFraction=None, symmetric=None, minWidth=True, slopeLength=None, maxSlope=None, slopeStartHeight=None):
        threshold = threshold if threshold is not None else self.config.get("threshold", 0)
        peakNo = peakNo if peakNo is not None else self.config.get("peakNo", 8)
        roi = roi if roi is not None else self.config.get("roi", [None,None])
        distanceFactor = distanceFactor if distanceFactor is not None else self.config.get("distanceFactor", 2)
        widthFraction = widthFraction if widthFraction is not None else self.config.get("widthFraction", 0.5)
        symmetric = symmetric if symmetric is not None else self.config.get("symmetric", True)
        slopeLength = slopeLength if slopeLength is not None else self.config.get("slopeLength", False)
        maxSlope = maxSlope if maxSlope is not None else self.config.get("maxSlope", False)

        
        peakFunc = partial(
            findPeaksInTrace_np,
            peakNo=peakNo,
            cutOff=threshold,
            widthFactor=distanceFactor,
            widthFraction=widthFraction,
            symmetric=symmetric,
            minWidth=minWidth,
            slopeLength=slopeLength,
            maxSlope=maxSlope,
            slopeStartHeight=slopeStartHeight
        )
    
        results_list = []
        for det in tqdm(range(self.data.sizes["detector"]), desc="Finding peaks in ToFs",position=2,leave=False,disable=True):
            sliceDet = self.data.isel(detector=det)
            sliceDet = sliceDet.isel(sample=slice(roi[0],roi[1]))
            
            sample_coords = sliceDet["sample"].values # the actual sample indices
            
            # Create peak function that converts array indices to actual sample coordinates
            def peakFuncWithCoords(trace, coords=sample_coords):
                peaks = findPeaksInTrace_np(trace, peakNo=peakNo, cutOff=threshold,
                                            widthFactor=distanceFactor,widthFraction=widthFraction, symmetric=symmetric,
                                            minWidth=minWidth)
                # replace array indices with real sample coordinates
                if peaks is not None and len(peaks) > 0:
                    indices = peaks[:, 0].astype(int)
                    # Clamp indices to valid range
                    indices = np.clip(indices, 0, len(coords) - 1)
                    peaks[:, 0] = coords[indices]
                return peaks
        
            results_det = xr.apply_ufunc(
                peakFunc,
                sliceDet,
                input_core_dims=[["sample"]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[object]
            )
            results_list.append(results_det)
    
        self.results = xr.concat(results_list, dim="detector")
        self.results = self.results.persist()
        self.dataframe()
        self.results["pos"] = (self.results["pos"]+roi[0]).astype(int) if roi[0] is not None else self.results["pos"].astype(int)
        return self


    def plot(self, trainIndex=None, pulseIndex=None, num=None, xmin=None, xmax=None, ymin=None, ymax=None, widthFraction=None,
              logScale=False, showGaussianFit=False, raw=False, savename=False):
        widthFraction = widthFraction if widthFraction is not None else self.config.get("widthFraction", 0.5)
        train_ids = self.data.indexes["pulse"].get_level_values("trainId")
        pulse_ids = self.data.indexes["pulse"].get_level_values("pulseId")
        

        randomTrainId = np.random.choice(train_ids)
        randomPulseId = np.random.choice(pulse_ids)
        
        
        trainId = trainIndex if trainIndex is not None else self.config.get("plotTrainIndex", randomTrainId)
        pulseId = pulseIndex if pulseIndex is not None else self.config.get("plotPulseIndex", randomPulseId)

        
        print(f"Random trainId: {trainId}, pulseId: {pulseId}")

        plotYNum = int(np.ceil(len(np.unique(self.data["detector"]))/4))
        fig, ax = plt.subplots(plotYNum,4,figsize=(12, 3*plotYNum),sharex='all', sharey='all')
        ax = ax.flatten()
        j=0
        if ymax is None:
            ymax = self.data.max() * 1.05
        
        # Define Paul Tol's bright color palette for colorblind-friendly plots
        tolBlue = '#4477AA'
        tolCyan = '#66CCEE'
        tolGreen = '#228833'
        tolYellow = '#CCBB44'
        tolRed = '#EE6677'
        tolPurple = '#AA3377'
        tolGrey = '#BBBBBB'
        tolBlack = '#000000'
        
        # Define colors for multiple Gaussians using Tol palette
        gaussColors = [tolRed, tolYellow, tolGrey, tolGreen]
        
        for ToF in self.data["detector"].to_index():
            trace = self.data.sel(detector=ToF,pulse={"trainId": trainId, "pulseId": pulseId})
            ax[j].set_title(f"ToF: {ToF}")
            ax[j].grid(True)
            ax[j].plot(trace,marker='.', color = tolBlue,  markersize=0 ,alpha=1,linewidth = 1)
            if logScale:
                #trace = np.clip(trace,1e-12,None)
                ax[j].set_yscale('symlog', linthresh=1e-2)
            ax[j].set_ylim([ymin, ymax])
            ax[j].set_xlim([xmin, xmax])
            if not isinstance(self.results, pd.DataFrame):
                print("Warning: results is not a DataFrame. Call .dataframe() first.")
                j += 1
                continue

            if not raw: 
                for peakNo in self.results["peakNo"].unique():
                    peak = self.results[(self.results["detector"]==ToF)&(self.results["peakNo"]==peakNo)&(self.results["trainId"]==trainId)&(self.results["pulseId"]==pulseId)]
                    if peak.empty:
                        continue
                    pos = peak["pos"].iloc[0]
                    height = peak["height"].iloc[0]
                    widthl = peak["width left"].iloc[0]
                    widthr = peak["width right"].iloc[0]
                    ax[j].hlines(y=height*widthFraction, xmin=pos+widthl, xmax=pos+widthr, colors=tolPurple)
                    ax[j].scatter(x=pos,y=height,color=tolPurple)
                    
                    # Plot baseline if available
                    if "baseline left" in peak.columns and "baseline right" in peak.columns:
                        baselineL = peak["baseline left"].iloc[0]+pos
                        baselineR = peak["baseline right"].iloc[0]+pos
                        if baselineL is not False and not pd.isna(baselineL) and not pd.isna(baselineR):
                            ax[j].plot([int(baselineL), int(baselineR)], [trace[int(baselineL)], trace[int(baselineR)]], color=tolGreen, linestyle='--')
                            baselineAdjustedTrace = trace.to_numpy().copy()

                            baselineSlope = (trace[int(baselineR)] - trace[int(baselineL)]) / (baselineR - baselineL)
                            offset = trace[int(baselineL)] - baselineSlope * baselineL
                            
                            for k in range(int(baselineL), int(baselineR) + 1):
                                baselineAdjustedTrace[k] = baselineAdjustedTrace[k] - (baselineSlope * k + offset)
                            
                            baselineX = np.arange(int(baselineL), int(baselineR)+1)
                            baselineAdjustedTrace = baselineAdjustedTrace[int(baselineL):int(baselineR)+1]
                            ax[j].plot(baselineX, baselineAdjustedTrace, color=tolGreen, linestyle='dotted', label='Baseline Adjusted' if peakNo == 0 else '',alpha=0.7)
                    
                    # Plot Gaussian fit if available and requested
                    if showGaussianFit:
                        # Check how many Gaussians were fitted for this peak
                        gaussCount = 0
                        for i in range(10):  # Check up to 10 Gaussians
                            suffix = f"_{i+1}" if i > 0 else ""
                            colName = f'gauss{suffix}_amplitude'
                            if colName in peak.columns and not pd.isna(peak[colName].iloc[0]):
                                gaussCount += 1
                            else:
                                break
                        
                        if gaussCount > 0:
                            # Determine common xFit range for all Gaussians
                            xRange = max(abs(widthl), abs(widthr)) * 3
                            xFit = np.linspace(pos - xRange, pos + xRange, 200)
                            totalYFit = np.zeros_like(xFit)
                            
                            # Plot each Gaussian
                            for i in range(gaussCount):
                                suffix = f"_{i+1}" if i > 0 else ""
                                amp = peak[f'gauss{suffix}_amplitude'].iloc[0]
                                center = peak[f'gauss{suffix}_center'].iloc[0]
                                sigma = peak[f'gauss{suffix}_sigma'].iloc[0]
                                
                                color = gaussColors[i % len(gaussColors)]
                                
                                # Calculate FWHM from Gaussian fit: FWHM = 2.355 * sigma
                                fwhmHalf = 1.177 * sigma
                                
                                # Generate Gaussian curve where amp is peak height
                                yFit = amp * np.exp(-0.5 * ((xFit - center) / sigma)**2)
                                
                                # FWHM line at half the peak height
                                ax[j].hlines(y=amp/2, xmin=center-fwhmHalf, xmax=center+fwhmHalf, 
                                        colors=color, linestyle="-", linewidth=2, alpha=0.7)
                                
                                ax[j].plot(xFit, yFit, color=color, linewidth=2, alpha=0.7, linestyle='--')
                                
                                # Accumulate for total curve
                                totalYFit += yFit
                            
                            # Plot sum of all Gaussians if multiple
                            if gaussCount > 1:
                                ax[j].plot(xFit, totalYFit, color=tolBlack, linewidth=2.5, linestyle='-', alpha=0.8)
            j+=1
        fig.supxlabel("Sample")
        fig.supylabel("Intensity")
        if savename is not False:
            plt.savefig(f"{savename}.png",dpi=600)
        return self
    
    def plotSingle(self, trainIndex=None, pulseIndex=None, ToF=0, xmin=None, xmax=None, ymin=None, ymax=None, figsize=(12, 8), savename=False, logScale=False, showGaussianFit=False):

        train_ids = self.data.indexes["pulse"].get_level_values("trainId")
        pulse_ids = self.data.indexes["pulse"].get_level_values("pulseId")
        
        randomTrainId = np.random.choice(train_ids)
        randomPulseId = np.random.choice(pulse_ids)
        
        trainId = trainIndex if trainIndex is not None else self.config.get("plotTrainIndex", randomTrainId)
        pulseId = pulseIndex if pulseIndex is not None else self.config.get("plotPulseIndex", randomPulseId)
        
        print(f"Plotting trainId: {trainId}, pulseId: {pulseId}, ToF: {ToF}")

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        if ymax is None:
            ymax = self.data.max() * 1.05
        
        # Define Paul Tol's bright color palette for colorblind-friendly plots
        tolBlue = '#4477AA'
        tolCyan = '#66CCEE'
        tolGreen = '#228833'
        tolYellow = '#CCBB44'
        tolRed = '#EE6677'
        tolPurple = '#AA3377'
        tolGrey = '#BBBBBB'
        tolBlack = '#000000'
        
        trace = self.data.sel(detector=ToF, pulse={"trainId": trainId, "pulseId": pulseId})
        sample_coords = self.data['sample'].values
        ax.set_title(f"ToF: {ToF}")
        ax.set_ylabel('Signal')
        ax.set_xlabel('Sample')
        ax.grid(True)
        ax.plot(trace, marker='.', color=tolBlue, markersize=0, alpha=1, linewidth=1, label='Data')
        
        if logScale:
            ax.set_yscale('symlog', linthresh=1e-2)
        
        ax.set_ylim([ymin, ymax])
        ax.set_xlim([xmin, xmax])
        
        if not isinstance(self.results, pd.DataFrame):
            print("Warning: results is not a DataFrame. Call .dataframe() first.")
            ax.legend()
            if savename is not False:
                plt.savefig(f"{savename}.png", dpi=600)
            return self
            
        for peakNo in self.results["peakNo"].unique():
            peak = self.results[(self.results["detector"]==ToF)&(self.results["peakNo"]==peakNo)&(self.results["trainId"]==trainId)&(self.results["pulseId"]==pulseId)]
            if peak.empty:
                continue
            pos = peak["pos"].iloc[0]
            height = peak["height"].iloc[0]
            widthl = peak["width left"].iloc[0]
            widthr = peak["width right"].iloc[0]
            baselineL = peak["baseline left"].iloc[0]
            baselineR = peak["baseline right"].iloc[0]
            ax.hlines(y=height/2, xmin=pos+widthl, xmax=pos+widthr, colors=tolRed, label='FWHM' if peakNo == 0 else '')
            ax.scatter(x=pos, y=height, color=tolRed, label='Peak' if peakNo == 0 else '')
            
            # Plot baseline if available
            if "baseline left" in peak.columns and "baseline right" in peak.columns:
                if baselineL is not False and not pd.isna(baselineL) and not pd.isna(baselineR):
                    baselineL += pos
                    baselineR += pos
                    ax.plot([int(baselineL), int(baselineR)], [trace[int(baselineL)], trace[int(baselineR)]], color=tolGreen, linestyle='--', label='Baseline' if peakNo == 0 else '')
                    baselineAdjustedTrace = trace.to_numpy().copy()
                    
                    # Compute linear baseline: y = slope * x + offset
                    baselineSlope = (trace[int(baselineR)] - trace[int(baselineL)]) / (baselineR - baselineL)
                    offset = trace[int(baselineL)] - baselineSlope * baselineL
                    
                    # Subtract baseline only in the region around the peak
                    for k in range(int(baselineL), int(baselineR) + 1):
                        baselineAdjustedTrace[k] = baselineAdjustedTrace[k] - (baselineSlope * k + offset)
                    
                    baselineX = np.arange(int(baselineL), int(baselineR)+1)
                    baselineAdjustedTrace = baselineAdjustedTrace[int(baselineL):int(baselineR)+1]
                    ax.plot(baselineX, baselineAdjustedTrace, color=tolGreen, linestyle='dotted', label='Baseline Adjusted' if peakNo == 0 else '',alpha=0.7)
            
            # Plot Gaussian fit if available and requested
            if showGaussianFit:
                # Define colors for multiple Gaussians using Tol palette
                gauss_colors = [tolPurple, tolCyan, tolYellow,tolGrey, tolGreen]
                
                # Check how many Gaussians were fitted for this peak
                gauss_count = 0
                for i in range(10):  # Check up to 10 Gaussians
                    suffix = f"_{i+1}" if i > 0 else ""
                    col_name = f'gauss{suffix}_amplitude'
                    if col_name in peak.columns and not pd.isna(peak[col_name].iloc[0]):
                        gauss_count += 1
                    else:
                        break
                
                if gauss_count > 0:
                    # Determine common x_fit range for all Gaussians
                    x_range = max(abs(widthl), abs(widthr)) * 3
                    x_fit = np.linspace(pos - x_range, pos + x_range, 200)
                    total_y_fit = np.zeros_like(x_fit)
                    
                    # Plot each Gaussian
                    for i in range(gauss_count):
                        suffix = f"_{i+1}" if i > 0 else ""
                        amp = peak[f'gauss{suffix}_amplitude'].iloc[0]
                        center = peak[f'gauss{suffix}_center'].iloc[0]
                        sigma = peak[f'gauss{suffix}_sigma'].iloc[0]
                        
                        color = gauss_colors[i % len(gauss_colors)]
                        
                        # Calculate FWHM from Gaussian fit: FWHM = 2.355 * sigma
                        fwhm_half = 1.177 * sigma
                        label_fwhm = f'Gauss {i+1} FWHM' if gauss_count > 1 else 'Gauss FWHM'
                        
                        # Generate Gaussian curve where amp is peak height
                        y_fit = amp * np.exp(-0.5 * ((x_fit - center) / sigma)**2)
                        
                        # FWHM line at half the peak height
                        ax.hlines(y=amp/2, xmin=center-fwhm_half, xmax=center+fwhm_half, 
                                 colors=color, linestyle="-", linewidth=2, alpha=0.7,
                                 label=label_fwhm if peakNo == 0 else '')
                        
                        label_fit = f'Gauss {i+1}' if gauss_count > 1 else 'Gaussian Fit'
                        ax.plot(x_fit, y_fit, color=color, linewidth=2, alpha=0.7, linestyle='--', 
                               label=label_fit if peakNo == 0 else '')
                        
                        # Accumulate for total curve
                        total_y_fit += y_fit
                    
                    # Plot sum of all Gaussians if multiple
                    if gauss_count > 1:
                        ax.plot(x_fit, total_y_fit, color=tolBlack, linewidth=2.5, linestyle='-', 
                               label='Total Fit' if peakNo == 0 else '', alpha=0.8)
                else:
                    if peakNo == 0:
                        print("Gaussian fit columns not found. Run .fitGaussians() first.")
        
        ax.legend()
        plt.savefig("traces.png", dpi=600)
        return self




    def dataframe(self):
        """
        Convert the list-of-arrays results from peak finding into a pandas DataFrame
        with columns ["detector","trainId","pulseId","peakNo","pos","height","width left","width right","fwhm area"].
        """
        all_results = []
        all_metadata = []
    
        # iterate over detectors
        for det_idx, det in enumerate(self.results["detector"].values):
            # get pulse MultiIndex
            pulse_index = self.results["pulse"].to_index()
            data_det = self.results.isel(detector=det_idx).values
    
            # iterate over pulses (still required because peaks per pulse vary)
            for (trainId, pulseId), peaks in zip(pulse_index, data_det):
                if peaks is None or len(peaks) == 0:
                    continue
                peaks = np.array(peaks)  # shape: (num_peaks, 5)
                peakNos = np.arange(len(peaks)).reshape(-1,1)
                all_results.append(np.hstack([peakNos, peaks]))
                # replicate metadata for each peak
                all_metadata.append(np.tile([det, trainId, pulseId], (len(peaks), 1)))
    
        if len(all_results) == 0:
            self.results = pd.DataFrame(
                columns=["detector","trainId","pulseId","peakNo","pos","height","width left","width right","fwhm area","baseline left","baseline right"]
            )
            return self
    
        # stack results vertically
        all_results = np.vstack(all_results)
        all_metadata = np.vstack(all_metadata)
    
        # create DataFrame
        df = pd.DataFrame(
            np.hstack([all_metadata, all_results]),
            columns=["detector","trainId","pulseId","peakNo","pos","height","width left","width right","fwhm area","baseline left","baseline right"]
        )
    
        # convert appropriate columns to int
        df[["detector","trainId","pulseId","peakNo"]] = df[["detector","trainId","pulseId","peakNo"]].astype(int)
    
        self.results = df
        return self

    def fitGaussians(self, useUpperHalf=True, roiWidthMultiplier=None, roiAbsolute=None, multiGauss=None, baseline=None):
        """
        Fit Gaussian functions to each detected peak.
        
        Parameters
        ----------
        useUpperHalf : bool, optional
            If True, only fit data above FWHM (half-height) to avoid background noise.
            Default is True.
        roiWidthMultiplier : float, optional
            Multiplier for the detected peak width to define the fitting ROI.
            For example, 2.0 means fit region extends 2x the peak width on each side.
            If None, uses 2.0 for useUpperHalf=True, 1.0 for useUpperHalf=False.
            For multi-Gaussian fits, consider using larger values (e.g., 4.0) to capture shoulders.
        roiAbsolute : tuple or list, optional
            Absolute ROI as (left_offset, right_offset) from peak center in sample units.
            If provided, overrides roiWidthMultiplier. Example: (-50, 50) for ±50 samples.
        multiGauss : list or array, optional
            Number of Gaussians to fit for each peakNo. 
            Example: [1, 2, 1] means fit 1 Gaussian for peak 0, 2 Gaussians for peak 1, 1 Gaussian for peak 2.
            If None, fits single Gaussian to all peaks.
        baseline : tuple or list, optional
            Baseline correction specified as two sample positions [left_sample, right_sample].
            A linear baseline is drawn between these two points and subtracted before fitting.
            Example: [100, 300] draws baseline from sample 100 to sample 300.
            Useful for peaks with shoulders or asymmetric background.
            
        Returns
        -------
        self : PeakFinder
            Returns self with Gaussian fit parameters added to self.results DataFrame.
            Adds columns: 'gauss_amplitude', 'gauss_center', 'gauss_sigma', 'gauss_area', 'gauss_fwhm_area'
            For multi-Gaussian fits, adds: 'gauss_1_amplitude', 'gauss_1_center', etc.
        """
        if not isinstance(self.results, pd.DataFrame):
            raise ValueError("Must call .dataframe() before fitting Gaussians")
        
        # Set default ROI multiplier
        if roiWidthMultiplier is None:
            roiWidthMultiplier = 2.0 if useUpperHalf else 1.0
        
        # Determine max number of Gaussians needed for column creation
        max_gaussians = 1
        if multiGauss is not None:
            max_gaussians = max(multiGauss)
        
        # Initialize columns for Gaussian parameters
        for i in range(max_gaussians):
            suffix = f"_{i+1}" if i > 0 else ""
            self.results[f'gauss{suffix}_amplitude'] = np.nan
            self.results[f'gauss{suffix}_center'] = np.nan
            self.results[f'gauss{suffix}_sigma'] = np.nan
            self.results[f'gauss{suffix}_area'] = np.nan
            self.results[f'gauss{suffix}_fwhm_area'] = np.nan
        self.results['gauss_fwhm_area'] = np.nan
        
        # Define Gaussian functions where amplitude = peak height
        def gaussian(x, amplitude, center, sigma):
            """Single Gaussian where amplitude is the peak height"""
            return amplitude * np.exp(-0.5 * ((x - center) / sigma)**2)
        
        def multi_gaussian(x, *params):
            """Sum of multiple Gaussians. Params: [amp1, center1, sigma1, amp2, center2, sigma2, ...]"""
            n_gaussians = len(params) // 3
            result = np.zeros_like(x, dtype=float)
            for i in range(n_gaussians):
                amp = params[i*3]
                center = params[i*3 + 1]
                sigma = params[i*3 + 2]
                result += amp * np.exp(-0.5 * ((x - center) / sigma)**2)
            return result
        
        # Iterate through each peak
        for idx, row in self.results.iterrows():
            try:
                # Determine number of Gaussians for this peak
                peak_no = int(row['peakNo'])
                if multiGauss is not None and peak_no < len(multiGauss):
                    n_gaussians = multiGauss[peak_no]
                else:
                    n_gaussians = 1
                
                # Get the trace for this peak
                trace = self.data.sel(
                    detector=row['detector'],
                    pulse={"trainId": row['trainId'], "pulseId": row['pulseId']}
                ).values
                
                sample_coords = self.data['sample'].values
                pos = int(row['pos'])
                height = row['height']
                width_left = int(row['width left'])
                width_right = int(row['width right'])
                
                # Find actual position in sample_coords array
                pos_idx = np.searchsorted(sample_coords, pos)
                
                # Determine fitting region
                if roiAbsolute is not None:
                    # Use absolute ROI offsets
                    left_offset = roiAbsolute[0]
                    right_offset = roiAbsolute[1]
                    fit_start = max(0, pos_idx + int(left_offset))
                    fit_end = min(len(trace), pos_idx + int(right_offset))
                else:
                    # Use width multiplier
                    fit_start = max(0, pos_idx + int(width_left * roiWidthMultiplier))
                    fit_end = min(len(trace), pos_idx + int(width_right * roiWidthMultiplier))
                
                # Apply baseline correction if specified
                trace_corrected = trace.copy()
                if baseline is not None:
                    baseline_left, baseline_right = baseline
                    # Find indices for baseline points
                    left_idx = np.searchsorted(sample_coords, baseline_left)
                    right_idx = np.searchsorted(sample_coords, baseline_right)
                    
                    # Get baseline values at the two points
                    left_val = trace[left_idx]
                    right_val = trace[right_idx]
                    
                    # Create linear baseline
                    baseline_slope = (right_val - left_val) / (sample_coords[right_idx] - sample_coords[left_idx])
                    baseline_line = left_val + baseline_slope * (sample_coords - sample_coords[left_idx])
                    
                    # Subtract baseline
                    trace_corrected = trace - baseline_line
                
                if useUpperHalf:
                    # Only use data above half-maximum
                    half_height = height / 2
                    
                    # Get data in the region
                    region_data = trace_corrected[fit_start:fit_end]
                    region_samples = sample_coords[fit_start:fit_end]
                    
                    # Only keep points above half-height
                    mask = region_data >= half_height
                    if np.sum(mask) < 3:  # Need at least 3 points to fit
                        continue
                    
                    x_data = region_samples[mask]
                    y_data = region_data[mask]
                else:
                    # Use full region data
                    x_data = sample_coords[fit_start:fit_end]
                    y_data = trace_corrected[fit_start:fit_end]
                
                if len(x_data) < 3 * n_gaussians:  # Need enough points for all Gaussians
                    continue
                
                # Create initial guess
                if n_gaussians == 1:
                    # Single Gaussian
                    p0 = [height, pos, abs(width_right - width_left) / 2.355]
                    popt, _ = curve_fit(gaussian, x_data, y_data, p0=p0, maxfev=5000)
                    params_list = [popt]
                else:
                    # Multiple Gaussians - distribute across the peak region
                    # For multi-Gaussian, use full region data (not upper half only)
                    x_data_full = sample_coords[fit_start:fit_end]
                    y_data_full = trace_corrected[fit_start:fit_end]
                    
                    p0 = []
                    bounds_lower = []
                    bounds_upper = []
                    
                    # Better sigma guess: divide peak width by number of Gaussians
                    sigma_guess = abs(width_right - width_left) / (2.355 * max(n_gaussians, 1))
                    region_width = x_data_full[-1] - x_data_full[0]
                    
                    # Improved amplitude guess: distribute amplitude based on n_gaussians
                    # For 3+ Gaussians, assume middle ones might be stronger
                    amp_guess = height / max(n_gaussians * 0.8, 1)
                    
                    for i in range(n_gaussians):
                        # Distribute centers evenly across the peak
                        if n_gaussians == 1:
                            center_offset = 0
                        else:
                            # Spread centers across the region
                            center_offset = (i - (n_gaussians-1)/2) * (region_width / (n_gaussians + 1))
                        
                        p0.extend([amp_guess, pos + center_offset, sigma_guess])
                        
                        # Set bounds for each Gaussian
                        # Amplitude: 0 to 2*height to allow flexibility
                        # Center: within the fitting region
                        # Sigma: minimum 10% of guess, max entire region
                        bounds_lower.extend([0, x_data_full[0], sigma_guess * 0.1])
                        bounds_upper.extend([height * 2.0, x_data_full[-1], abs(region_width)])
                    
                    # Fit multiple Gaussians with bounds
                    try:
                        popt, _ = curve_fit(multi_gaussian, x_data_full, y_data_full, 
                                           p0=p0, bounds=(bounds_lower, bounds_upper), maxfev=15000)
                    except:
                        # If bounded fit fails, try without bounds
                        try:
                            popt, _ = curve_fit(multi_gaussian, x_data_full, y_data_full, 
                                               p0=p0, maxfev=15000)
                        except:
                            # If all else fails, skip this peak
                            continue
                    
                    # Split parameters into separate Gaussians
                    params_list = []
                    for i in range(n_gaussians):
                        params_list.append(popt[i*3:(i+1)*3])
                
                # Store fit parameters for each Gaussian
                from scipy.special import erf
                for i, params in enumerate(params_list):
                    suffix = f"_{i+1}" if i > 0 else ""
                    amp, center, sigma = params
                    
                    self.results.at[idx, f'gauss{suffix}_amplitude'] = amp
                    self.results.at[idx, f'gauss{suffix}_center'] = center
                    self.results.at[idx, f'gauss{suffix}_sigma'] = sigma
                    
                    # Calculate Gaussian area: for exp(-0.5*(x-c)^2/s^2), area = amp * sigma * sqrt(2*pi)
                    gauss_area = amp * sigma * np.sqrt(2 * np.pi)
                    self.results.at[idx, f'gauss{suffix}_area'] = gauss_area
                    
                    # Calculate area within FWHM boundaries
                    fwhm_half = 1.177 * sigma
                    erf_arg = fwhm_half / (sigma * np.sqrt(2))
                    gauss_fwhm_area = amp * sigma * np.sqrt(2 * np.pi) * erf(erf_arg)
                    self.results.at[idx, f'gauss{suffix}_fwhm_area'] = gauss_fwhm_area
                
            except Exception as e:
                # If fit fails, leave as NaN
                continue
        
        return self

class AuxFunc:
    def __init__(self, data):
        self.data = data

    def addData(self, moreData, key="Photon Energy", axis="trainId"):
        if axis not in self.data.columns:
            raise KeyError(f"{axis} not found in self.data")
        if key not in moreData.columns:
            raise KeyError(f"{key} not found in moreData")

        mapping = moreData.set_index(axis)[key]
        self.data[key] = self.data[axis].map(mapping)
        return self

def streamXarray(data):
    for trainId, group in data.groupby("trainId"):
        #time.sleep(0.01)
        yield group

class PhotonEnergyProcessor(Configurable):
    def __init__(self, proposal, runNo, loaderClass, config=None):
        super().__init__(config)
        self.loaderClass = loaderClass
        self.proposal = proposal
        self.runNo = runNo
        self.data = None
        self.photonEnergies = None
        self.firstTrainId = int
        self.results = []
        self.run = self.loaderClass(self.proposal, self.runNo)
        self.pf = PeakFinder(self.data,config=self.config)


    def getRunEnergies(self,singleRun=None, energyStart=None, energyStop=None, energyStep=None):
        singleRun = singleRun if singleRun is not None else self.config.get("singleRun", True)
        energyStart = energyStart if energyStart is not None else self.config.get("energyStart", 0)
        energyStop = energyStop if energyStop is not None else self.config.get("energyStop", None)
        energyStep = energyStep if energyStep is not None else self.config.get("energyStep", 1)
        if singleRun:
            self.run.load(key="Photon Energy",trainStart=None, trainStop=None, trainStep=None, pulseStart=None, pulseStop=None, pulseStep=None)
            self.firstTrainId = self.run.photonEnergy.trainId[0]
            self.photonEnergies = self.run.photonEnergy.groupby("Photon Energy",as_index=False).last()
        else:
            self.run = self.loaderClass(self.proposal, self.runNo,config=self.config)
            self.run.load(key="Photon Energy",trainStart=None, trainStop=None, trainStep=None, pulseStart=None, pulseStop=None, pulseStep=None)
            self.photonEnergies = self.run.photonEnergy#.groupby("Photon Energy",as_index=False))
            #self.photonEnergies["daq_run"] = self.photonEnergies["daq_run"].astype(int)
            #self.photonEnergies["trainId"] = self.photonEnergies["trainId"].astype(int)
        return self

        trainStart = trainStart or self.config.get("trainStart", None)
        trainStop = trainStop or self.config.get("trainStop", None)
        trainStep = trainStep or self.config.get("trainStep", None)
        pulseStart = pulseStart or self.config.get("pulseStart", None)
        pulseStop = pulseStop or self.config.get("pulseStop", None)
        pulseStop = pulseStep or self.config.get("pulseStep", None)
    
    def processEnergies(self, energyStart=None, energyStop=None, energyStep=None, trainSliceStop=None, singleRun=None, peakFinderConfig=None):
        energyStart = energyStart if energyStart is not None else self.config.get("energyStart", 0)
        energyStop = energyStop if energyStop is not None else self.config.get("energyStop", None)
        energyStep = energyStep if energyStep is not None else self.config.get("energyStep", 1)
        singleRun = singleRun if singleRun is not None else self.config.get("singleRun", True)

        #loaderConfig = loaderConfig or self.config.get("loaderConfig", {})
        #peakFinderConfig = peakFinderConfig or self.config.get("PeakFinder", {})
        #print("Start processing...")
        if energyStop == None:
            energyStop = len(self.photonEnergies)-1

        peakChunks = []
        for i in tqdm(np.arange(energyStart, energyStop, energyStep),desc="Processing trains",position=0):
            if singleRun:
                trainStart = self.photonEnergies.trainId[i] - self.firstTrainId
                trainSliceStop = self.run.config.get("trainStep",1)
                self.data = self.run.load(trainStart = trainStart, trainStop = int(trainStart+trainSliceStop)).defaultPreprocessing().data
            else:
                self.run = self.loaderClass(self.proposal, self.runNo[0], config=self.config)
                self.data = self.run.load().defaultPreprocessing().data
            peakChunk = PeakFinder(self.data,config=self.config).stack().normalize().process().dataframe().results
            AuxFunc(peakChunk).addData(self.run.photonEnergy)
            peakChunks.append(peakChunk)
        self.results = pd.concat(peakChunks)
        print("Done!")
        return self.results

class Calibrate(Configurable):
    def __init__(self, results, config=None):
        super().__init__(config)
        self.results = results
        self.energyParam = []
        self.transmissionParam = []


    def madFilter(self, x, y, thresh=3):
        y_np = np.asarray(y)
        med = np.median(y_np)
        mad = np.median(np.abs(y_np - med))
    
        if mad == 0:
            return np.ones_like(y_np, dtype=bool)
    
        z = 0.6745 * (y_np - med) / mad
        return np.abs(z) <= thresh

        
    def energy(self,relPos=False,peakNo=None,guess=None):
        peakNo = peakNo if peakNo is not None else self.config.get("peakNo", 0)
        guess = (guess or self.config.get("initial guess", None))  # Will be computed from data if None
        avgPos = self.results.groupby(["detector","peakNo","Photon Energy"])["pos"].mean().reset_index()
        energyParam = []
        transmissionParam = []
        for det in avgPos["detector"].unique():
            pos = pd.DataFrame(avgPos[(avgPos["detector"]==det)&(avgPos["peakNo"]==peakNo)]["pos"]).reset_index()["pos"]
            if relPos:
                pos0 = pd.DataFrame(avgPos[(avgPos["detector"]==det)&(avgPos["peakNo"]==0)]["pos"]).reset_index()
                pos = pos - pos0["pos"]
            energy = avgPos[(avgPos["detector"]==det)&(avgPos["peakNo"]==peakNo)]["Photon Energy"]
            if len(energy)<3:
                continue
            xdata = pos.values
            ydata = energy.values
            goodData = self.madFilter(xdata,ydata)
            
            # Use provided guess or compute data-driven initial guesses
            if guess is None:
                # Create data-driven initial guesses for energyCalibFunc: p0 + p1/sqrt(e+p2) + p3/(e+p4)^(3/2)
                y_min, y_max = min(ydata), max(ydata)
                x_min, x_max = min(xdata), max(xdata)
                y_range = y_max - y_min
                x_mean = np.mean(xdata)
                
                # p0: baseline energy (slightly below minimum)
                p0_guess = y_min - 0.1 * y_range
                
                # p1, p3: scale with energy range and typical positions
                # The 1/sqrt term typically dominates, give it more weight
                p1_guess = y_range * np.sqrt(x_mean)
                p3_guess = y_range * (x_mean)**(1.5) * 0.1
                
                # p2, p4: offset to ensure positive arguments, related to position scale
                # Need e + p2 > 0, so p2 > -min(xdata)
                p2_guess = abs(x_min) + x_mean
                p4_guess = abs(x_min) + x_mean
                
                guess_to_use = [p0_guess, p1_guess, p2_guess, p3_guess, p4_guess]
                
                # Set bounds to help convergence
                # p0: allow some range around the baseline
                # p1, p3: must be positive (contribute positive energy)
                # p2, p4: must keep arguments positive, i.e., > -x_min
                bounds_lower = [
                    y_min - y_range,           # p0: below minimum energy
                    0,                          # p1: positive
                    -x_min + 1e-6,             # p2: keep x+p2 > 0
                    0,                          # p3: positive  
                    -x_min + 1e-6              # p4: keep x+p4 > 0
                ]
                bounds_upper = [
                    y_max + y_range,           # p0: above maximum energy
                    y_range * x_max * 10,      # p1: large but not unbounded
                    x_max * 10,                # p2: reasonable upper limit
                    y_range * (x_max**1.5),    # p3: large but not unbounded
                    x_max * 10                 # p4: reasonable upper limit
                ]
            else:
                # Use provided guess
                guess_to_use = guess
                bounds_lower = -np.inf
                bounds_upper = np.inf
            
            try:
                params, pcov = curve_fit(energyCalibFunc, xdata[goodData], ydata[goodData], 
                                        p0=guess_to_use, bounds=(bounds_lower, bounds_upper), maxfev=1000000)
                p0Fit, p1Fit, p2Fit, p3Fit, p4Fit = params
                perr = np.sqrt(np.diag(pcov))
                energyParam.append({"detector": det, "peakNo": peakNo, "p0": p0Fit, "p1": p1Fit, "p2": p2Fit, "p3": p3Fit, "p4": p4Fit, "p0 error":perr[0], "p1 error":perr[1], "p2 error":perr[2], "p3 error":perr[3], "p4 error":perr[4]})
            except RuntimeError:
                # Fit failed
                energyParam.append({
                    'detector': det,
                    'peakNo': peakNo,
                    'p0': np.nan,
                    'p1': np.nan,
                    'p2': np.nan,
                    'p3': np.nan,
                    'p4': np.nan,
                    "p0 error":np.nan,
                    "p1 error":np.nan,
                    "p2 error":np.nan,
                    "p3 error":np.nan,
                    "p4 error":np.nan,
            })
        self.energyParam = pd.DataFrame(energyParam)
        return self

    def transmission(self, peakNo=None, setBeta=None, setPhi=None, setPlin=None,intMethod="height"):
        transmissionParam = []
        """
        beta = beta or self.config.get("beta",0)
        peakNo = peakNo or self.config.get("Transmission PeakNo",0)
        """
        for energy in self.results["Photon Energy"].unique():
            for ToF in self.results["detector"].unique():
                selData = self.results[(self.results["peakNo"]==peakNo)&(self.results["Photon Energy"]==energy)&(self.results["detector"]==ToF)]
                pos = selData["pos"].mean().astype(int)
                trace = selData[intMethod].mean()
                theta = np.deg2rad(selData["Angles"].to_numpy())
                g = polarization_model(theta, Plin=setPlin, phi=setPhi,beta2=setBeta)
                transPar = g/trace
                transmissionParam.append({"detector": ToF, "Photon Energy": energy,"sample": pos, "Transmission Coefficient": transPar[0]})
        self.transmissionParam = pd.DataFrame(transmissionParam)
        return self

    def plotTransmission(self,ymin=None,ymax=None):
        plotYNum = int(np.ceil(self.results["detector"].nunique()/4))
        fig, ax = plt.subplots(plotYNum,4,figsize=(12, 3*plotYNum),sharex='all', sharey='all')
        plt.ylabel ('Transmission Coefficient')
        plt.xlabel ('Photon Energy')
        ax = ax.flatten()
        j=0
        for ToF in self.transmissionParam["detector"].unique():
            xdata = self.transmissionParam[(self.transmissionParam["detector"]==ToF)]["Photon Energy"]
            ydata = self.transmissionParam[(self.transmissionParam["detector"]==ToF)]["Transmission Coefficient"]
            ax[j].set_title(f"ToF: {ToF}")
            ax[j].grid(True)
            ax[j].plot(xdata,ydata,marker='.', color = 'teal',  markersize=2 ,alpha=1,linewidth = 0)
            ax[j].set_ylim([ymin, ymax])
            j+=1
        plt.savefig("Transmission.png",dpi=600)
        return self
                

    def plotEnergy(self, peakNo = None, plotReg = True, relPos=False, ymin=None, ymax=None, xmin=None, xmax=None):
        peakNo = peakNo if peakNo is not None else self.config.get("peakNo", 0)
        plotYNum = int(np.ceil(self.results["detector"].nunique()/4))
        fig, ax = plt.subplots(plotYNum,4,figsize=(12, 3*plotYNum),sharex='all', sharey='all')


        ax = ax.flatten()
        j=0
            
        for i in self.results["detector"].unique():
            pos = self.results[(self.results["detector"]==i)&(self.results["peakNo"]==peakNo)]["pos"].reset_index()
            if relPos:
                pos0 = pd.DataFrame(self.results[(self.results["detector"]==i)&(self.results["peakNo"]==0)]["pos"]).reset_index()
                pos = pos - pos0
                        
            xdata = pos["pos"]
            ydata = self.results[(self.results["peakNo"]==peakNo)&(self.results["detector"]==i)]["Photon Energy"]
            
            if plotReg:
                goodData = self.madFilter(xdata,ydata)
                xFit = np.linspace(xdata[goodData].min(),xdata[goodData].max(),500)
                xFitExt = np.linspace(xdata[goodData].min(),xdata.max(),500)
                #print(xdata[goodData].min(),xdata[goodData].max())
                p0,p1,p2,p3,p4 = self.energyParam.loc[self.energyParam["detector"] == i, ["p0", "p1", "p2", "p3", "p4"]].values[0]
                ax[j].plot(xFitExt, energyCalibFunc(xFitExt, p0, p1, p2, p3, p4), color="salmon",linestyle="dashed", label="Fit", linewidth=0.9)
                ax[j].plot(xFit, energyCalibFunc(xFit, p0, p1, p2, p3, p4), color="forestgreen", label="Fit")
                ax[j].set_xlim([xmin, xmax])
                ax[j].set_ylim([ymin, ymax])
            ax[j].set_title(f"ToF: {i}")
            ax[j].grid(True)
            ax[j].plot(xdata,ydata,marker='.', color = 'teal',  markersize=2 ,alpha=1,linewidth = 0)
            j+=1
        fig.supxlabel("Sample")
        fig.supylabel("Photon Energy")
        plt.savefig("Energy.png",dpi=600)
        plt.show()
        return self


class Fitter(Configurable):
    def __init__(self, results, config=None):
        super().__init__(config)
        self.results = results
        ToFs = self.results["detector"].unique()
        params = pd.DataFrame(columns=["detector","Photon Energy","Transmission Coefficient"],index=ToFs)
        params["detector"] = ToFs
        params["Photon Energy"] = self.results["Photon Energy"]
        params["Transmission Coefficient"] = [1]*len(ToFs)
        self.params = params


    def pol(self, transParam=None, peakNo=None, beta=0, setPlin=None, fitBeta=False, intMethod="height", plot=True):
        peakNo = peakNo if peakNo is not None else self.config.get("peakNo", 0)
        transParam = transParam if transParam is not None else self.params
        fullTheta = np.linspace(0,2*np.pi,16,endpoint=False)
        area = self.results[self.results["peakNo"]==peakNo][["fwhm area","height","detector","Angles"]]
        calib = transParam
        calibArea = pd.merge(area,calib,on="detector")
        calibArea["calibValue"] = calibArea[intMethod] * calibArea["Transmission Coefficient"] #/ max(calibArea[intMethod])
            
        theta = calibArea["Angles"].values*np.pi/180
        trace = calibArea["calibValue"]
        maxTrace = max(trace)#[0]

        fit_kws = dict(method='trf', ftol=1e-9, xtol=1e-9, gtol=1e-9, maxfev=100000)

        if setPlin is not None:
            # --- Plin is fixed; fit phi (and optionally beta2) ---
            beta0 = beta if beta != 0 else 1.0
            if fitBeta:
                def model(theta, phi, beta2, scale):
                    return polarization_model(theta, Plin=setPlin, phi=phi, beta2=beta2, scale=scale)
                p0     = [0.0, beta0, 1.0]
                bounds = ([-np.pi, -4.0, 0.0], [np.pi, 4.0, 10.0])
                popt, pcov = curve_fit(model, theta, trace, p0=p0, bounds=bounds, **fit_kws)
                phi_rad_fit, beta2_fit, scale_fit = popt
                sigma_phi = np.rad2deg(np.sqrt(pcov[0, 0]))
                sigma_beta2 = np.sqrt(pcov[1, 1])
                sigma_scale = np.sqrt(pcov[2, 2])
            else:
                def model(theta, phi, scale):
                    return polarization_model(theta, Plin=setPlin, phi=phi, beta2=beta, scale=scale)
                p0     = [0.0, 1.0]
                bounds = ([-np.pi, 0.0], [np.pi, 10.0])
                popt, pcov = curve_fit(model, theta, trace, p0=p0, bounds=bounds, **fit_kws)
                phi_rad_fit, scale_fit = popt
                beta2_fit = beta
                sigma_phi = np.rad2deg(np.sqrt(pcov[0, 0]))
                sigma_beta2 = 0.0
                sigma_scale = np.sqrt(pcov[1, 1])

            Plin_fit  = setPlin
            phi_fit   = np.rad2deg(phi_rad_fit)
            sigma_P   = 0.0  # Plin was fixed

        else:
            # --- A/B parameterisation (A = Plin·cos2φ, B = Plin·sin2φ) ---
            if fitBeta:
                beta0 = beta if beta != 0 else 1.0
                def model(theta, A, B, beta2, scale):
                    return sepModel(theta, A, B, beta2=beta2, scale=scale)
                p0     = [0.0, 0.0, beta0, 1.0]
                bounds = ([-2.0, -2*np.pi, -4.0, 0.0], [2.0, 2*np.pi, 4.0, 10.0])
                popt, pcov = curve_fit(model, theta, trace, p0=p0, bounds=bounds, **fit_kws)
                A_fit, B_fit, beta2_fit, scale_fit = popt
                cov_AB = pcov[:2, :2]
                sigma_beta2 = np.sqrt(pcov[2, 2])
                sigma_scale = np.sqrt(pcov[3, 3])
            else:
                def model(theta, A, B, scale):
                    return sepModel(theta, A, B, beta2=beta, scale=scale)
                p0     = [0.0, 0.0, 1.0]
                bounds = ([-2.0, -2*np.pi, 0.0], [2.0, 2*np.pi, 10.0])
                popt, pcov = curve_fit(model, theta, trace, p0=p0, bounds=bounds, **fit_kws)
                A_fit, B_fit, scale_fit = popt
                beta2_fit = beta
                cov_AB = pcov[:2, :2]
                sigma_beta2 = 0.0
                sigma_scale = 0.0

            Plin_fit = np.sqrt(A_fit**2 + B_fit**2)
            phi_rad_fit = 0.5 * np.arctan2(B_fit, A_fit)
            phi_fit  = np.rad2deg(phi_rad_fit)

            sigma_A2 = cov_AB[0, 0]
            sigma_B2 = cov_AB[1, 1]
            sigma_AB = cov_AB[0, 1]
            P = Plin_fit
            sigma_P = np.sqrt(
                (A_fit/P)**2 * sigma_A2 +
                (B_fit/P)**2 * sigma_B2 +
                2*(A_fit*B_fit/P**2)*sigma_AB
            )
            sigma_phi = np.rad2deg(0.5 * np.sqrt(
                (B_fit**2 * sigma_A2 + A_fit**2 * sigma_B2 - 2*A_fit*B_fit*sigma_AB)
                / (A_fit**2 + B_fit**2)**2
            ))

        # Round for display / return
        Plin_fit  = np.round(Plin_fit,  8)
        phi_fit   = np.round(phi_fit,   8)
        scale_fit = np.round(scale_fit, 8)
        beta2_fit = np.round(beta2_fit, 8)

        if fitBeta:
            fit_params = [Plin_fit, phi_fit, scale_fit, beta2_fit]
            errors     = [sigma_P, sigma_phi, sigma_scale, sigma_beta2]
        else:
            fit_params = [Plin_fit, phi_fit, scale_fit]
            errors     = [sigma_P, sigma_phi, 0]

        theta_fit     = np.linspace(0, 2*np.pi, 360)
        intensity_fit = polarization_model(theta_fit, Plin_fit, phi_rad_fit, beta2_fit, scale_fit)

        if plot:
            fig, ax = plt.subplots(figsize=(6,4), subplot_kw={'projection': 'polar'})
            ax.plot(theta, trace, marker="o", linewidth=0, label='Data')
            if Plin_fit > 0.015:
                ax.plot([phi_rad_fit, phi_rad_fit], [0, maxTrace], color="orange")
                ax.plot([phi_rad_fit + np.pi, phi_rad_fit + np.pi], [0, maxTrace], color="orange")
            label = f"Fitted Plin: {Plin_fit:.5f}"
            if fitBeta:
                label += f", beta: {beta2_fit:.4f}"
            ax.plot(theta_fit, intensity_fit, label=label, color="green")
            ax.set_yticks([])
            ax.set_theta_zero_location("N")  # 0° at top
            ax.set_theta_direction(-1)       # clockwise
            ax.legend(loc="lower right")
            plt.savefig("pol.png", dpi=600)
            plt.show()
            
        return fit_params, errors


        

def polarization_model(theta, Plin=1, phi=0, beta2=2, scale=1):
    return scale*(1 + (beta2 / 4) * (1 + 3 * Plin * np.cos(2 * (theta - phi))))

def sepModel(theta, A, B,beta2=2, scale=1):
    return scale * (1 + (beta2/4)*(1 + 3*(A*np.cos(2*theta) + B*np.sin(2*theta))))


class Plotter(Configurable):
    """
    Polar plotter for ToF detector data.

    Displays intensity as a function of sample position (radius) and detector
    angle (from config) using a viridis colorscale.  Optionally overlays the
    polarization_model function for a given set of Plin / phi / beta2 / scale.

    Angle and ToF-channel mapping are read from the NXSLoader section of the
    global config (keys ``ToF`` and ``angles``), or can be supplied directly
    via *config*.

    Parameters
    ----------
    data : xarray.DataArray
        Data with dimensions ``detector``, ``pulse`` (multi-index trainId /
        pulseId) and ``sample``.
    config : dict, optional
        Override keys from the global config.
    """

    CONFIG_KEY = "NXSLoader"

    def __init__(self, data, results=None, config=None):
        super().__init__(config)
        self.data = data
        self.results = results

        # Build detector -> angle mapping (degrees) from config
        fullAngles = np.array(
            self.config.get("angles", np.linspace(0, 360, 16, endpoint=False))
        )
        detectors = self.config.get("ToF", list(range(15)))

        if len(fullAngles) == len(detectors):
            # angles list corresponds 1-to-1 with the selected detectors
            self.detectorAngles = {d: a for d, a in zip(detectors, fullAngles)}
        else:
            # angles list is a full set — index into it by detector number
            self.detectorAngles = {d: fullAngles[d] for d in detectors}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _selectSlice(self, trainId=None, pulseIndex=0, pulseId=None):
        """Return a 2-D (detector × sample) DataArray for one train/pulse."""
        index = self.data["pulse"].to_index()
        availableTrains = index.get_level_values("trainId").unique()

        if trainId is None:
            trainId = availableTrains[0]

        if pulseId is None:
            trainPulses = index[index.get_level_values("trainId") == trainId]
            pulseId = trainPulses.get_level_values("pulseId")[pulseIndex]

        return (
            self.data.sel(pulse={"trainId": trainId, "pulseId": pulseId}),
            trainId,
            pulseId,
        )

    def _buildIntensityGrid(self, traces, anglesRad, nTheta=720):
        """
        Interpolate *traces* (n_det × n_sample) onto a uniform theta grid.

        Returns
        -------
        intensityGrid : ndarray, shape (nTheta, nSample)
        thetaGrid     : ndarray, shape (nTheta,)
        """
        nSamples = traces.shape[1]
        thetaGrid = np.linspace(0, 2 * np.pi, nTheta, endpoint=False)
        intensityGrid = np.zeros((nTheta, nSamples))

        sortIdx = np.argsort(anglesRad)
        sortedAngles = anglesRad[sortIdx]
        sortedTraces = traces[sortIdx]

        for si in range(nSamples):
            vals = sortedTraces[:, si]
            # Extend arrays for periodic (wrap-around) interpolation
            xpExt = np.concatenate(
                [sortedAngles - 2 * np.pi, sortedAngles, sortedAngles + 2 * np.pi]
            )
            fpExt = np.concatenate([vals, vals, vals])
            intensityGrid[:, si] = np.interp(thetaGrid, xpExt, fpExt)

        return intensityGrid, thetaGrid

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plot(
        self,
        trainId=None,
        pulseIndex=0,
        pulseId=None,
        sampleMin=None,
        sampleMax=None,
        vMin=None,
        vMax=None,
        figsize=(8, 8),
        interpolate=True,
        nTheta=720,
        transParam=None,
        showModel=False,
        Plin=1.0,
        phi=0.0,
        beta2=2.0,
        scale=1.0,
        modelRadius=None,
        modelColor="red",
        modelLabel=None,
        cbarLabel="Intensity",
        title=None,
        ax=None,
        direction=1,
        orientation = "E"
    ):
        """
        Plot data in polar coordinates.

        Parameters
        ----------
        trainId : int or None
            Train ID to display.  ``None`` → first available train.
        pulseIndex : int
            Pulse index within the train (used when *pulseId* is ``None``).
        pulseId : int or None
            Explicit pulse ID.  Overrides *pulseIndex* when given.
        sampleMin, sampleMax : int or None
            Range of sample indices to display (inner / outer radius).
        vMin, vMax : float or None
            Colorscale limits.  ``None`` → auto.
        figsize : tuple
            Figure size (ignored when *ax* is provided).
        interpolate : bool
            ``True`` – smooth 2-D interpolation between detector angles.
            ``False`` – each detector drawn as a narrow discrete wedge.
        nTheta : int
            Number of theta steps used for the interpolated grid.
        transParam : pandas.DataFrame or None
            Calibration table with at least columns ``detector`` and
            ``Transmission Coefficient`` (same format as used by
            ``Fitter.pol()``).  Each detector's trace is multiplied by its
            coefficient before plotting.  Detectors not listed in the table
            are left unscaled.
        showModel : bool
            Overlay ``polarization_model`` as a radial curve.
        Plin, phi, beta2, scale : float
            Parameters forwarded to ``polarization_model``.
        modelRadius : float or None
            Peak radius of the overlaid model curve.
            Defaults to ``sampleMax``.
        modelColor : str
            Line colour for the model overlay.
        modelLabel : str or None
            Legend label for the model overlay.
        cbarLabel : str
            Label for the colourbar.
        title : str or None
            Plot title.  ``None`` → auto-generated from trainId / pulseId.
        ax : matplotlib.axes.Axes or None
            Existing polar axes to draw into.  When ``None`` a new figure is
            created.

        Returns
        -------
        self
        """
        dataSlice, trainId, pulseId = self._selectSlice(trainId, pulseIndex, pulseId)

        # Keep only detectors that have an angle mapping
        availableDets = [
            int(d) for d in dataSlice.detector.values if int(d) in self.detectorAngles
        ]
        availableDets = sorted(availableDets, key=lambda d: self.detectorAngles[d])

        anglesRad = np.array(
            [self.detectorAngles[d] * np.pi / 180 for d in availableDets]
        )

        # Build (n_det × n_sample) intensity array
        traces = np.array(
            [dataSlice.sel(detector=d).values for d in availableDets]
        )

        # Apply calibration (Transmission Coefficient) if provided
        if transParam is not None:
            calibMap = (
                transParam
                .set_index("detector")["Transmission Coefficient"]
                .to_dict()
            )
            for i, det in enumerate(availableDets):
                coeff = calibMap.get(det, calibMap.get(str(det), 1.0))
                traces[i] = traces[i] * coeff



        # Sample coordinate values (may be offset by ROI)
        sampleCoords = dataSlice.coords["sample"].values
        idxMin = 0 if sampleMin is None else int(np.searchsorted(sampleCoords, sampleMin))
        idxMax = len(sampleCoords) if sampleMax is None else int(
            np.searchsorted(sampleCoords, sampleMax, side="right")
        )

        traces = traces[:, idxMin:idxMax]
        rVals = sampleCoords[idxMin:idxMax]

        # Radius bin edges for pcolormesh
        if len(rVals) > 1:
            dr = rVals[1] - rVals[0]
        else:
            dr = 1.0
        rEdges = np.concatenate([[rVals[0] - dr / 2], rVals + dr / 2])

        # ------------------------------------------------------------------
        # Set up axes
        # ------------------------------------------------------------------
        ownFig = ax is None
        if ownFig:
            fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})

        vmin = vMin if vMin is not None else traces.min()
        vmax = vMax if vMax is not None else traces.max()

        # ------------------------------------------------------------------
        # Draw data
        # ------------------------------------------------------------------
        if interpolate:
            intensityGrid, thetaGrid = self._buildIntensityGrid(
                traces, anglesRad, nTheta=nTheta
            )
            dTheta = thetaGrid[1] - thetaGrid[0]
            thetaEdges = np.append(thetaGrid - dTheta / 2, thetaGrid[-1] + dTheta / 2)

            # pcolormesh(theta, r, C) → C shape (n_r, n_theta)
            mesh = ax.pcolormesh(
                thetaEdges,
                rEdges,
                intensityGrid.T,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                shading="auto",
            )
        else:
            # Discrete wedge per detector — width = half gap to each neighbour
            nDets = len(anglesRad)
            if nDets > 1:
                # anglesRad is already sorted by angle
                gaps = np.diff(anglesRad, append=anglesRad[0] + 2 * np.pi)
                wedgeLeft = np.roll(gaps, 1) / 2
                wedgeRight = gaps / 2
            else:
                wedgeLeft = np.array([np.pi])
                wedgeRight = np.array([np.pi])

            for i, det in enumerate(availableDets):
                ang = anglesRad[i]
                thetaEdges = np.array([ang - wedgeLeft[i], ang + wedgeRight[i]])
                C = traces[i, :][np.newaxis, :]   # shape (1, n_sample)
                ax.pcolormesh(
                    thetaEdges,
                    rEdges,
                    C.T,
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax,
                    shading="auto",
                )

            # Dummy mesh for colourbar
            mesh = ax.pcolormesh(
                [0, 0.01], [rEdges[0], rEdges[-1]], [[vmin]], cmap="viridis",
                vmin=vmin, vmax=vmax, shading="auto"
            )
            mesh.set_visible(False)

        # Colourbar
        if ownFig:
            plt.colorbar(mesh, ax=ax, label=cbarLabel, shrink=0.7, pad=0.1)

        # ------------------------------------------------------------------
        # Overlay polarization model
        # ------------------------------------------------------------------
        if self.results is not None:
            for i in self.results["peakNo"].unique():
                selPeak = self.results[self.results["peakNo"]==i]
                peakPos = selPeak["pos"]
                peakAngles = selPeak["Angles"]
                widthL = selPeak["width left"]
                widthR = selPeak["width right"]
                ax.scatter(np.deg2rad(peakAngles), peakPos, color=modelColor, marker="o")
                for angle in peakAngles:
                    ax.plot([np.deg2rad(angle), np.deg2rad(angle)], [peakPos + widthL, peakPos + widthR], color=modelColor, linewidth=0.5, linestyle="-")
        # ------------------------------------------------------------------
        # Cosmetics
        # ------------------------------------------------------------------
        ax.set_theta_zero_location(orientation)
        ax.set_theta_direction(direction)

        # Constrain radial axis to the selected sample range
        ax.set_rlim(rEdges[0], rEdges[-1])

        # Remove outer border of the polar plot
        ax.spines["polar"].set_visible(False)
        
        # Place r tick labels in a gap between detectors when possible.
        # All 16 possible slots are spaced 22.5° apart; find the midpoint of the
        # largest empty arc so the labels don't overlap with data wedges.
        # Only do this for the discrete (non-interpolated) mode and when at
        # least one slot is missing.
        allSlotAngles = np.linspace(0, 2 * np.pi, 16, endpoint=False)  # 16 slots
        usedAnglesRad = np.array(sorted(anglesRad % (2 * np.pi)))

        def _largestGapMidpoint(usedAngles, allSlots):
            """Return the midpoint (rad) of the largest arc not covered by usedAngles."""
            # Snap each slot to "used" or "empty"
            usedSet = set(
                np.argmin(np.abs(allSlots[:, None] - usedAngles[None, :]), axis=0)
            )
            emptySlots = [a for i, a in enumerate(allSlots) if i not in usedSet]
            if not emptySlots:
                return None
            # Find largest consecutive run of empty slots (circular)
            emptyIdx = sorted(
                [i for i, a in enumerate(allSlots) if i not in usedSet]
            )
            # Build runs with wrap-around
            n = len(allSlots)
            bestLen, bestStart = 0, emptyIdx[0]
            runLen, runStart = 1, emptyIdx[0]
            for k in range(1, len(emptyIdx) + len(emptyIdx)):
                cur = emptyIdx[k % len(emptyIdx)]
                prev = emptyIdx[(k - 1) % len(emptyIdx)]
                if (cur - prev) % n == 1:
                    runLen += 1
                else:
                    runLen, runStart = 1, cur
                if runLen > bestLen:
                    bestLen, bestStart = runLen, runStart
                if k >= len(emptyIdx) - 1 and runLen == bestLen:
                    break
            midSlot = (bestStart + bestLen // 2) % n
            return float(np.degrees(allSlots[midSlot]))

        if not interpolate and len(usedAnglesRad) < len(allSlotAngles):
            gapDeg = _largestGapMidpoint(usedAnglesRad, allSlotAngles)
            if gapDeg is not None:
                ax.set_rlabel_position(gapDeg)
        # else: leave at matplotlib default (22.5°)

        # Re-apply white background after rlabel position may have changed
        for label in ax.get_yticklabels():
            label.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1))

        # Tick marks at detector angles
        ax.set_xticks(anglesRad+anglesRad[1]/2)
        ax.set_xticklabels(
            [f"{self.detectorAngles[d]:.1f}°\nToF {d}" for d in availableDets], fontsize=12
        )
        ax.tick_params(axis="x", pad=15)
        for label in ax.get_yticklabels():
            label.set_backgroundcolor((1, 1, 1, 0.7))
            label.set_fontsize(12)

        plotTitle = title

        ax.set_title(plotTitle, pad=15)
        plt.savefig("polHeat.png",dpi=600)
        if ownFig:
            plt.tight_layout()
            plt.show()
        
        return self


#Main Functions
#determines peakwidth of symetric peaks
def findSymmetricPeakWidth(trace,peak):
    '''
    Function to calculate the fwhm of a peak within a trace
    assuming the peak is symmetric.
    
    Parameters
    --------
    trace, array:
        trace with the peak
    peak, int:
        index of the peak which width shall be calculated.

    Returns
    --------
    peakWidth, int:
        half width at half maximum.
    '''
    peakWidth = 0
    maxWidth=20
    while peakWidth < maxWidth and (peak+peakWidth) < len(trace):
        if trace[peak]/2 <= trace[peak+peakWidth]:
            peakWidth +=1
        else:
            break
    
    return -peakWidth, peakWidth

#determines peakwidth of asymetric peaks
def findAsymmetricPeakWidth(trace,peak):
    '''
    Function to calculate the fwhm of a peak within a trace
    assuming the peak is asymmetric.
    
    Parameters
    --------
    trace, array:
        trace with the peak
    peak, int:
        index of the peak which width shall be calculated.

    Returns
    --------
    peakWidthL, int:
        width at half maximum left of the peak. (negative value)
    peakWidthR, int:
        width at half maximum right of the peak.
    '''
    peakWidthR = 0
    maxWidth = 20
    while peakWidthR < maxWidth and (peak+peakWidthR) < len(trace):
        if trace[peak]/2 <= trace[peak+peakWidthR]:
            peakWidthR +=1
        else:
            break
            
    peakWidthL = -peakWidthR
    if trace[peak]/2 <= trace[peak+peakWidthL]:
        for i in range(20):
            if trace[peak]/2 <= trace[peak+peakWidthL]:
                peakWidthL -=1
            else:
                break

    if trace[peak]/2 >= trace[peak+peakWidthL]:
        for i in range(20):
            if trace[peak]/2 >= trace[peak+peakWidthL]:
                peakWidthL +=1
            else:
                break          
    return peakWidthL, peakWidthR

def findPeak(trace, widthFactor=2, symmetric = False):
    '''
    Function to find the minimum in a trace.

    Parameters
    --------
    trace, 1D array
        Array with the trace

    Returns
    --------
    trace, 1D array
        Array of the trace without the peak
    peak, int
        Index of the peak
    '''
    trace = trace.copy()
    peak = trace.argmax()
    trace, baselinePointL, baselinePointR = findPeakBaseline(trace,peak)
    # Apply baseline subtraction (findPeakBaseline now only finds the points)
    if baselinePointL is not False and baselinePointR is not False:
        blSlope = (trace[baselinePointR] - trace[baselinePointL]) / (baselinePointR - baselinePointL)
        blOffset = trace[baselinePointL] - blSlope * baselinePointL
        for k in range(baselinePointL, baselinePointR + 1):
            trace[k] = trace[k] - (blSlope * k + blOffset)
    height = trace.max()
 
    if symmetric == True:
        peakWidthL, peakWidthR = findSymmetricPeakWidth(trace,peak)
    else:
        peakWidthL, peakWidthR = findAsymmetricPeakWidth(trace,peak)
        
    trace[peak+(peakWidthL*widthFactor):peak+(peakWidthR*widthFactor)] = 0
    return trace, peak, height, peakWidthL, peakWidthR, baselinePointL, baselinePointR

def findPeakBaseline(trace, peak, slopeLength=5, maxSlope=4, startOffsetL=0, startOffsetR=0):
    """
    Find baseline endpoint indices around a peak by walking the slope.
    
    Walks left and right from the peak until the slope drops below maxSlope.
    Returns the trace (unmodified) and the two baseline endpoint indices.
    The caller is responsible for computing and subtracting the baseline.
    
    Parameters
    ----------
    startOffsetL : int
        Negative offset from peak to start the left walk (e.g. from FWHM left edge).
    startOffsetR : int
        Positive offset from peak to start the right walk (e.g. from FWHM right edge).
    """
    n = len(trace)
    
    # Walk left to find baseline start point
    i = startOffsetL  # start from peak (0) or from FWHM left edge (negative)
    baselinePointL = max(0, peak + startOffsetL)
    while i > -100:
        idxCurrent = peak + i
        idxLeft = peak - slopeLength + i
        # Bounds check
        if idxLeft < 0 or idxCurrent < 0:
            break
        slope = trace[idxCurrent] - trace[idxLeft]
        if slope <= maxSlope:
            break
        baselinePointL = idxLeft
        i -= 1
    
    # Walk right to find baseline end point
    j = startOffsetR  # start from peak (0) or from FWHM right edge (positive)
    baselinePointR = min(n - 1, peak + startOffsetR)
    while j < 100:
        idxCurrent = peak + j
        idxRight = peak + slopeLength + j
        # Bounds check
        if idxRight >= n or idxCurrent >= n:
            break
        slope = trace[idxRight] - trace[idxCurrent]
        if -slope <= maxSlope:
            break
        baselinePointR = idxRight
        j += 1
    
    # Use the actual baseline points for slope calculation
    idxL = baselinePointL
    idxR = baselinePointR
    
    # If baseline points are the same, no baseline found
    if idxR == idxL:
        return trace, False, False
    
    return trace, int(baselinePointL-peak), int(baselinePointR-peak)

def findPeaksInTrace(trace, peakNo , cutOff = -100, widthFactor=2, widthFraction=0.5, symmetric = True):
    results = []
    traceCopy = trace.copy()
    if traceCopy.max() > cutOff:
        for i in range(peakNo+1):
            if traceCopy.max() > cutOff:
                traceCopy, pos, height, widthL, widthR, baselinePointL, baselinePointR = findPeak(traceCopy, widthFactor=widthFactor,widthFraction=widthFraction, symmetric = symmetric)
                a = trace[pos+widthL:pos+widthR].sum()
                results.append([pos,height,widthL,widthR,a, baselinePointL, baselinePointR])
    if len(results) == peakNo+1:
        results.sort()
        resultsDicts = [{"pos": p, "height": h, "width left": wl, "width right":wr, "fwhm area": a, "baseline left": bl, "baseline right": br} for p, h, wl, wr, a, bl, br in results]
    else:
        resultsDicts = None
    return resultsDicts

def findPeak_np(trace, widthFactor=2, symmetric=False, maxWidth=20, minWidth=False, widthFraction=0.5, slopeLength=False, maxSlope=False, originalTrace=None, slopeStartHeight=None):
    """
    Find the largest peak in a trace and compute FWHM widths.
    Uses originalTrace (unzeroed) for baseline detection and peak analysis.
    The zeroed 'trace' is only used for iterative peak finding (argmax) and zeroing.
    
    Parameters
    ----------
    slopeStartHeight : float or None
        Fraction of peak height at which the baseline slope walk starts.
        E.g. 0.5 starts at FWHM edges, 0.8 starts where trace is at 80% of peak height.
        None (default) starts the walk from the peak center.
    Returns trace with peak zeroed, peak position, height, left width, right width.
    """
    trace = trace.copy()
    peak = trace.argmax()
    
    # Build the analysis trace: baseline-subtracted version of the original
    if originalTrace is not None:
        analysisTrace = originalTrace.copy()
    else:
        analysisTrace = trace.copy()
    
    if slopeLength or maxSlope is not False:
        # If slopeStartHeight is set, find where the trace crosses that fraction of peak height
        startOffsetL = 0
        startOffsetR = 0
        if slopeStartHeight is not None:
            rawHeight = analysisTrace[peak]
            if rawHeight > 0:
                threshold = rawHeight * slopeStartHeight
                left_sl = analysisTrace[max(0, peak-maxWidth):peak+1][::-1]
                right_sl = analysisTrace[peak:peak+maxWidth+1]
                prelim_wR = np.argmax(right_sl < threshold)
                if prelim_wR == 0 and right_sl[0] >= threshold:
                    prelim_wR = min(maxWidth, len(right_sl)-1)
                prelim_wL = -np.argmax(left_sl < threshold)
                if prelim_wL == 0 and left_sl[0] >= threshold:
                    prelim_wL = -min(maxWidth, len(left_sl)-1)
                startOffsetL = prelim_wL  # negative
                startOffsetR = prelim_wR  # positive
        
        # Use the original unzeroed trace for baseline slope analysis
        _, baseL, baseR = findPeakBaseline(analysisTrace, peak, slopeLength=slopeLength, maxSlope=maxSlope, startOffsetL=startOffsetL, startOffsetR=startOffsetR)
        # Apply baseline subtraction to the analysis trace
        if baseL is not False and baseR is not False:
            baselineL = peak + baseL
            baselineR = peak + baseR
            blSlope = (analysisTrace[baselineR] - analysisTrace[baselineL]) / (baselineR - baselineL)
            blOffset = analysisTrace[baselineL] - blSlope * baselineL
            for k in range(baselineL, baselineR + 1):
                analysisTrace[k] = analysisTrace[k] - (blSlope * k + blOffset)
    else:
        baselineL = None
        baselineR = None
    
    # Compute height and FWHM from the clean baseline-adjusted trace
    height = analysisTrace[peak]

    # Slice around peak (from analysis trace)
    left_slice = analysisTrace[max(0, peak-maxWidth):peak+1][::-1]  # reverse for left
    right_slice = analysisTrace[peak:peak+maxWidth+1]

    # Find first index below half max
    widthR = np.argmax(right_slice < height*widthFraction)
    if widthR == 0 and right_slice[0] >= height*widthFraction:
        widthR = min(maxWidth, len(right_slice)-1)

    widthL = -np.argmax(left_slice < height*widthFraction)
    if widthL == 0 and left_slice[0] >= height*widthFraction:
        widthL = -min(maxWidth, len(left_slice)-1)

    # For symmetric peaks, just use max of L/R
    if symmetric:
        w = max(abs(widthL), widthR)
        widthL, widthR = -w, w

    # Zero out peak region on the working trace (for iterative peak finding)
    start_zero = max(0, peak + widthL*widthFactor)
    stop_zero = min(len(trace), peak + widthR*widthFactor)
    if minWidth:
        min_width = min(abs(widthL), abs(widthR))
        widthL, widthR = -min_width, min_width

    # Area under peak (from analysis trace)
    start = max(0, peak + widthL)
    stop = min(len(trace), peak + widthR)
    area = analysisTrace[start:stop].sum()
    trace[start_zero:stop_zero] = 0

    return trace, peak, height, widthL, widthR, area, baseL, baseR

def findPeaksInTrace_np(trace, peakNo, cutOff=-100, widthFactor=2,widthFraction=0.5, symmetric=True, maxWidth=30, minWidth=False,slopeLength=5, maxSlope=4, slopeStartHeight=None):
    results = []

    originalTrace = trace.copy()  # Keep intact copy for baseline detection
    traceCopy = trace.copy()      # Working copy that gets zeroed for peak finding
    for _ in range(peakNo+1):
        if traceCopy.max() <= cutOff:
            break
        traceCopy, pos, height, widthL, widthR, area, baselineL, baselineR = findPeak_np(
            traceCopy, widthFactor=widthFactor, widthFraction=widthFraction, symmetric=symmetric, maxWidth=maxWidth,minWidth=minWidth,slopeLength=slopeLength, maxSlope=maxSlope, originalTrace=originalTrace, slopeStartHeight=slopeStartHeight)

        results.append([pos, height, widthL, widthR, area, baselineL, baselineR])

    if len(results) == peakNo + 1:
        results_arr = np.array(results, dtype=float)
        sorted_indices = np.argsort(results_arr[:, 0])  # sort by pos
        return results_arr[sorted_indices]
    
    return None  # Explicitly return None if no peaks found

"""
def findPeaksInTrace_sp(trace, peakNo, cutOff=0, widthFactor=2, symmetric=False, maxWidth=20):
    results = []

    traceCopy = trace.copy()
    pos, optRes = find_peaks(traceCopy, height=cutOff)

    i=0
    if len(pos)==peakNo+1:
        for peak in pos:
            
            height = optRes['peak_heights'][i]
            leftSlice = trace[max(0, peak-maxWidth):peak+1][::-1]  # reverse for left
            rightSlice = trace[peak:peak+maxWidth+1]
            
            widthR = np.argmax(rightSlice < height/2)
            if widthR == 0 and rightSlice[0] >= height/2:
                widthR = min(maxWidth, len(rightSlice)-1)
            
            widthL = -np.argmax(leftSlice < height/2)
            if widthL == 0 and leftSlice[0] >= height/2:
                widthL = -min(maxWidth, len(leftSlice)-1)
            widthL = -min(abs(widthL),abs(widthR))
            widthR = min(abs(widthL),abs(widthR))
            area = trace[peak+widthL:peak+widthR].sum()
            i+=1
        
            results.append([peak, height, widthL, widthR, area, baselineL, baselineR])

        results_arr = np.array(results, dtype=float)
        sorted_indices = np.argsort(results_arr[:, 0])  # sort by pos
        return results_arr[sorted_indices]
"""

def energyCalibFunc(e, p0, p1, p2, p3, p4):
    return p0 + (p1/((e+p2)**(1/2))) + (p3/((e+p4)**(3/2)))