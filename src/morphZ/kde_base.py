"""
kde_base.py -- Shared base class for KDE implementations with JSON bandwidth loading support.
"""
import json
import os
from typing import Union, Dict, Any


class KDEBase:
    """
    Base class for KDE implementations with JSON bandwidth loading support.
    """

    def _load_bandwidths_from_json(self, json_path: str) -> Dict[str, float]:
        """
        Load bandwidth values from a JSON file created by compute_and_save_bandwidths.

        The function now returns the same grouped format that gets saved to JSON:
        [["param1", "param2", 0.1], ["param3", 0.2]]

        Parameters
        ----------
        json_path : str
            Path to the JSON file containing bandwidth information

        Returns
        -------
        Dict[str, float]
            Dictionary mapping parameter names to bandwidth values
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Bandwidth JSON file not found: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # The JSON file may be one of the following formats:
        # 1) Grouped list format (legacy from earlier versions):
        #    [["p1","p2", factor], ["p3", factor], ...]
        # 2) Structured dict format mirroring selected_pairs/groups files:
        #    {"pairs"|"groups": [{"names":[...], "factor": f}],
        #     "singles": [{"name": "p", "factor": f}]}
        # 3) Legacy dict mapping param->factor or {"bandwidths": {...}}
        if isinstance(data, list):
            bandwidths = {}
            for group in data:
                if not isinstance(group, list) or len(group) < 2:
                    continue
                param_names = group[:-1]
                bw_value = group[-1]
                for param_name in param_names:
                    if isinstance(param_name, str):
                        bandwidths[param_name] = float(bw_value)
            return bandwidths

        if isinstance(data, dict):
            # Structured dict with pairs/groups and singles
            out: Dict[str, float] = {}
            if "pairs" in data or "groups" in data:
                items = data.get("pairs", []) + data.get("groups", [])
                for item in items:
                    names = item.get("names") if isinstance(item, dict) else None
                    fac = None
                    if isinstance(item, dict):
                        # prefer 'bw'; fallback to 'factor' for backward compatibility
                        fac = item.get("bw", item.get("factor"))
                    if isinstance(names, (list, tuple)) and fac is not None:
                        for n in names:
                            if isinstance(n, str):
                                out[n] = float(fac)
                singles = data.get("singles", [])
                for s in singles:
                    if isinstance(s, dict):
                        name = s.get("name")
                        fac = s.get("bw", s.get("factor"))
                        if isinstance(name, str) and fac is not None:
                            out[name] = float(fac)
                if out:
                    return out
            # Legacy dictionary formats
            if "bandwidths" in data and isinstance(data["bandwidths"], dict):
                return {str(k): float(v) for k, v in data["bandwidths"].items()}
            # Assume direct param->factor mapping
            if all(isinstance(v, (int, float)) for v in data.values()):
                return {str(k): float(v) for k, v in data.items()}
            raise ValueError(f"Invalid JSON structure in {json_path}. Unsupported dict layout.")

        else:
            raise ValueError(f"Invalid JSON structure in {json_path}. Expected list or dict format.")

    def _prepare_bandwidth_dict(self, kde_bw: Union[str, float, Dict[str, float]],
                              bw_json_path: Union[str, None],
                              param_names: list) -> Dict[str, float]:
        """
        Prepare bandwidth dictionary, combining JSON values with user overrides.

        Parameters
        ----------
        kde_bw : Union[str, float, Dict[str, float]]
            User-provided bandwidth specification
        bw_json_path : Union[str, None]
            Path to JSON file with bandwidth values
        param_names : list
            List of parameter names

        Returns
        -------
        Dict[str, float]
            Dictionary mapping parameter names to bandwidth values
        """
        # Start with JSON bandwidths if provided
        bandwidths = {}
        if bw_json_path is not None:
            json_bandwidths = self._load_bandwidths_from_json(bw_json_path)
            bandwidths.update(json_bandwidths)

        # Apply user overrides
        if isinstance(kde_bw, dict):
            bandwidths.update(kde_bw)
        elif isinstance(kde_bw, (str, float)):
            # For string/float, it will be applied directly in the KDE fitting
            pass

        return bandwidths if bandwidths else None

    def _get_bandwidth_for_params(self, param_names: list, bandwidth_dict: Union[Dict[str, float], None],
                                default_bw: Union[str, float,]= "silverman") -> Union[str, float, list]:
        """
        Get bandwidth for a group of parameters, using available values from bandwidth_dict
        and default_bw for missing parameters.

        Parameters
        ----------
        param_names : list
            List of parameter names
        bandwidth_dict : Union[Dict[str, float], None]
            Dictionary mapping parameter names to bandwidth values
        default_bw : Union[str, float]
            Default bandwidth method/value to use for missing parameters

        Returns
        -------
        Union[str, float, list]
            Bandwidth specification for the parameters
        """
        if bandwidth_dict is None or not bandwidth_dict:
            # No bandwidth dict or empty dict - use default for all parameters
            if len(param_names) == 1:
                return default_bw
            else:
                return default_bw

        # For single parameter, return specific bandwidth if available, otherwise default
        if len(param_names) == 1:
            return bandwidth_dict.get(param_names[0], default_bw)

        # For multiple parameters, we need to handle the case where gaussian_kde
        # requires either all numeric values or a single string method
        result = []
        all_numeric = True

        for name in param_names:
            if name in bandwidth_dict:
                bw_val = bandwidth_dict[name]
                result.append(bw_val)
                if not isinstance(bw_val, (int, float)):
                    all_numeric = False
            else:
                result.append(default_bw)
                if not isinstance(default_bw, (int, float)):
                    all_numeric = False

        # If all values are numeric, return the list; otherwise use the default method
        if all_numeric:
            return result
        else:
            # Mixed types or non-numeric - use the default method for all
            return default_bw
