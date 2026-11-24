"""Bind GT and read-only prediction XML sources to a flat list of images."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import cache, partial
from pathlib import Path


def same_xml_path(first, second):
    """Compare destinations even if missing, symlinked, or hard-linked."""
    first, second = Path(first), Path(second)
    if first.resolve() == second.resolve():
        return True
    try:
        return first.samefile(second)
    except FileNotFoundError:
        return False


@dataclass(frozen=True)
class XmlPair:
    gt: Path
    prediction: Path


def _matching_paths(directory, name):
    return [p for p in directory.iterdir() if p.name.casefold() == name.casefold()]


def _one_match(matches, fallback):
    if len(matches) > 1:
        raise ValueError(f"{fallback.name} 有多份同名 XML，请明确选择对应文件。")
    return matches[0] if matches else fallback


@dataclass(frozen=True)
class _Source:
    directory: Path
    bound_file: Path | None = None

    @classmethod
    def from_path(cls, path):
        if not path:
            return None
        path = Path(path).absolute()
        if path.is_dir():
            return cls(path)
        if path.suffix.casefold() == ".xml" and path.is_file():
            return cls(path.parent, path)
        raise ValueError(f"请选择已有的 XML 文件或目录：{path}")

    def match(self, name, is_selected, lookup):
        if is_selected and self.bound_file is not None:
            return self.bound_file
        return _one_match(lookup(self.directory, name), self.directory / name)

    def to_config(self):
        if self.bound_file:
            return {"file": str(self.bound_file)}
        return {"directory": str(self.directory)}


def _config_path(value):
    if not isinstance(value, str) or not value.strip() or not Path(value).is_absolute():
        raise ValueError("工程中的对照来源需要完整的绝对路径。")
    return Path(value)


def _restore_source(value):
    if value is None:
        return None
    if isinstance(value, dict) and set(value) == {"file"}:
        path = _config_path(value["file"])
        if path.is_file() and path.suffix.casefold() == ".xml":
            return path
    elif isinstance(value, dict) and set(value) == {"directory"}:
        path = _config_path(value["directory"])
        if path.is_dir():
            return path
    raise ValueError("工程中的对照 XML 来源无效或已移动，请重新选择文件或目录。")


def _nearby_matches(root, name, lookup):
    folders = [p for p in lookup(root, "xml") if p.is_dir()]
    if len(folders) > 1:
        raise ValueError("图片旁有多个名为 xml 的目录，请明确选择对应目录。")
    nested = folders[0] if folders else root / "xml"
    candidates = list(lookup(root, name))
    if folders:
        candidates.extend(lookup(nested, name))
    return candidates, nested / name


def _nearby_path(image, name, explicit, lookup):
    # Prefer xml/ when creating a new GT, as in single-source mode.
    candidates, fallback = _nearby_matches(image.parent, name, lookup)
    candidates = [p for p in candidates if p != explicit]
    if same_xml_path(fallback, explicit):
        fallback = image.parent / name
    return _one_match(candidates, fallback)


class ComparisonSources:
    def __init__(self, selected_image, gt_source=None, prediction_source=None, *, images):
        self.selected_image = Path(selected_image).resolve()
        self._gt = _Source.from_path(gt_source)
        self._prediction = _Source.from_path(prediction_source)
        if self._gt is None and self._prediction is None:
            raise ValueError("请至少指定一个 XML 文件或目录，以区分 GT 和预测。")
        self._images = {Path(image).resolve() for image in images}
        if self.selected_image not in self._images:
            raise ValueError("最初绑定 XML 的图片不在当前图片列表中。")
        self._stems = Counter(image.stem.casefold() for image in self._images)

    def to_config(self):
        return {
            "selected_image": str(self.selected_image),
            "gt": self._gt.to_config() if self._gt else None,
            "prediction": self._prediction.to_config() if self._prediction else None,
        }

    @classmethod
    def from_config(cls, config, *, images):
        if not isinstance(config, dict) or set(config) != {"selected_image", "gt", "prediction"}:
            raise ValueError("工程中的 XML 对照配置不完整。")
        selected = _config_path(config["selected_image"])
        return cls(selected, _restore_source(config["gt"]),
                   _restore_source(config["prediction"]), images=images)

    def resolve(self, image):
        """Resolve fresh on navigation and before writing. Missing layers have a path."""
        return self._resolve(image, _matching_paths)

    def scan_resolver(self):
        """Return a resolver with directory-name indexes local to one read-only scan."""
        @cache
        def index(directory):
            names = defaultdict(list)
            for path in directory.iterdir():
                key = path.name.casefold()
                if key == "xml" or key.endswith(".xml"):
                    names[key].append(path)
            return names

        def lookup(directory, name):
            return index(directory).get(name.casefold(), ())

        return partial(self._resolve, lookup=lookup)

    def _resolve(self, image, lookup):
        image = Path(image).resolve()
        if image not in self._images:
            raise ValueError("图片不在当前 XML 对照列表中。")
        if self._stems[image.stem.casefold()] > 1:
            raise ValueError(f"{image.stem} 有多张同名图片，无法确定对应 XML。")
        name = image.with_suffix(".xml").name
        is_selected = image == self.selected_image
        gt = self._gt.match(name, is_selected, lookup) if self._gt else None
        prediction = self._prediction.match(name, is_selected, lookup) if self._prediction else None
        if is_selected and self._gt and self._gt.bound_file and not gt.is_file():
            raise ValueError(f"指定的 GT 文件已移动，请重新选择：{gt}")
        if gt is None:
            gt = _nearby_path(image, name, prediction, lookup)
        if prediction is None:
            prediction = _nearby_path(image, name, gt, lookup)
        if same_xml_path(gt, prediction):
            raise ValueError("GT 与预测指向同一个 XML。请选择不同的文件或目录。")
        if not is_selected:
            for source in (self._gt, self._prediction):
                if source and source.bound_file and any(
                    same_xml_path(path, source.bound_file) for path in (gt, prediction)
                ):
                    raise ValueError(f"{source.bound_file.name} 已绑定到 {self.selected_image.name}。")
        # A custom GT filename or symlink can target another image's prediction.
        for path in {gt, gt.resolve()}:
            stem = path.stem.casefold()
            if stem == image.stem.casefold() or stem not in self._stems:
                continue
            if self._prediction:
                candidates = [self._prediction.match(path.name, False, lookup)]
            else:
                candidates, _ = _nearby_matches(image.parent, path.name, lookup)
            if any(same_xml_path(gt, p) for p in candidates):
                raise ValueError(f"{gt.name} 与另一张图片的预测 XML 冲突。")
        return XmlPair(gt, prediction)
