"""Bind GT and read-only prediction XML sources to images or a directory tree."""

from collections import defaultdict
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


def _file_id(path):
    try:
        info = path.stat()
    except FileNotFoundError:
        return None
    return info.st_dev, info.st_ino


@dataclass(frozen=True)
class XmlPair:
    gt: Path
    prediction: Path


def _matching_paths(directory, name):
    return [p for p in directory.iterdir() if p.name.casefold() == name.casefold()]


def _one_match(matches):
    if len(matches) > 1:
        raise ValueError(f"{matches[0].name} 有多份同名 XML，请明确选择对应文件。")
    return matches[0]


def _scan_lookup():
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

    return lookup


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


def _nearby_candidates(image, name, explicit, lookup):
    # Prefer xml/ when creating a new GT, as in single-source mode.
    candidates, fallback = _nearby_matches(image.parent, name, lookup)
    candidates = [p for p in candidates if p != explicit]
    if same_xml_path(fallback, explicit):
        fallback = image.parent / name
    return candidates or [fallback]


class ComparisonSources:
    def __init__(self, selected_image, gt_source=None, prediction_source=None, *, images, image_root=None):
        self.selected_image = Path(selected_image).resolve()
        self._gt = _Source.from_path(gt_source)
        self._prediction = _Source.from_path(prediction_source)
        if self._gt is None and self._prediction is None:
            raise ValueError("请至少指定一个 XML 文件或目录，以区分可编辑标签和只读对照标签。")
        self._images = {Path(image).resolve() for image in images}
        if self.selected_image not in self._images:
            raise ValueError("最初绑定 XML 的图片不在当前图片列表中。")
        self.image_root = Path(image_root).resolve(strict=True) if image_root is not None else None
        if self.image_root and (not self.image_root.is_dir() or
                                any(not img.is_relative_to(self.image_root) for img in self._images)):
            raise ValueError("对照图片必须位于所选图片目录中。")
        self._by_stem = defaultdict(list)
        self._by_location = defaultdict(list)
        for img in self._images:
            stem = img.stem.casefold()
            self._by_stem[stem].append(img)
            self._by_location[img.parent, stem].append(img)

    def source_paths_for(self, image):
        image = Path(image).resolve()
        if image not in self._images:
            raise ValueError("图片不在当前 XML 对照列表中。")

        def source_path(source):
            if source is None:
                return None
            if source.bound_file and image == self.selected_image:
                return source.bound_file
            return source.directory

        return source_path(self._gt), source_path(self._prediction)

    def to_config(self):
        config = {
            "selected_image": str(self.selected_image),
            "gt": self._gt.to_config() if self._gt else None,
            "prediction": self._prediction.to_config() if self._prediction else None,
        }
        if self.image_root is not None:
            config["image_root"] = str(self.image_root)
        return config

    @classmethod
    def from_config(cls, config, *, images):
        required = {"selected_image", "gt", "prediction"}
        if not isinstance(config, dict) or set(config) not in (required, required | {"image_root"}):
            raise ValueError("工程中的 XML 对照配置不完整。")
        selected = _config_path(config["selected_image"])
        root = _config_path(config["image_root"]) if "image_root" in config else None
        return cls(selected, _restore_source(config["gt"]),
                   _restore_source(config["prediction"]), images=images, image_root=root)

    def resolve(self, image):
        """Resolve fresh on navigation and before writing. Missing layers have a path."""
        return self._resolve(image, _matching_paths)

    def scan_resolver(self):
        """Return a resolver with directory-name indexes local to one read-only scan."""
        return partial(self._resolve, lookup=_scan_lookup())

    def export_resolver(self, layers):
        """Read the requested sources. Explicit single layers have independent matching."""
        layers = tuple(layers)
        if not layers or set(layers) - {"gt", "prediction"}:
            raise ValueError("请选择要导出的 XML 来源。")
        lookup = _scan_lookup()
        sources = {"gt": self._gt, "prediction": self._prediction}
        if len(layers) == 1 and sources[layers[0]] is not None:
            return partial(self._read_explicit_layer, layer=layers[0], source=sources[layers[0]],
                           lookup=lookup)

        def resolve(image):
            pair = self._resolve(image, lookup)
            return {layer: getattr(pair, layer) for layer in layers}

        return resolve

    def _read_explicit_layer(self, image, *, layer, source, lookup):
        image = self._checked_image(image)
        path = self._explicit_path(source, image, lookup)
        self._check_file_binding(image, (path,))
        return {layer: path}

    def _explicit_path(self, source, image, lookup):
        path = _one_match(self._explicit_candidates(source, image, lookup))
        if source is self._gt and image == self.selected_image and source.bound_file and not path.is_file():
            raise ValueError(f"指定的可编辑标签文件已移动，请重新选择：{path}")
        return path

    def _check_file_binding(self, image, paths):
        if image == self.selected_image:
            return
        for source in (self._gt, self._prediction):
            if source and source.bound_file and any(same_xml_path(p, source.bound_file) for p in paths):
                raise ValueError(f"{source.bound_file.name} 已绑定到 {self.selected_image.name}。")

    def _explicit_candidates(self, source, image, lookup, *, protect=False):
        if source.bound_file and image == self.selected_image:
            return [source.bound_file]
        name = image.with_suffix(".xml").name
        relative_parent = image.parent.relative_to(self.image_root) if self.image_root else Path()
        mirror = source.directory / relative_parent / name
        if mirror.parent == source.directory:
            return lookup(source.directory, name) or [mirror]
        try:
            matches = lookup(mirror.parent, name)
        except FileNotFoundError:
            matches = ()
        if matches:
            return matches
        flat = lookup(source.directory, name)
        if not flat:
            if protect and not self._by_location.get((self.image_root, image.stem.casefold())):
                # Saving a previously missing flat file could make it a prediction.
                return [mirror, source.directory / name]
            return [mirror]
        stem = image.stem.casefold()
        if len(self._by_stem[stem]) == 1:
            return flat
        if self._by_location.get((self.image_root, stem)):
            # The flat XML is the exact mirrored location of a root-level image.
            return [mirror]
        if protect:
            return list(flat) + [mirror]
        raise ValueError(f"{name} 对应多张同名图片，请按图片相对目录存放 XML。")

    def _prediction_candidates(self, image, lookup):
        if self._prediction:
            return self._explicit_candidates(self._prediction, image, lookup, protect=True)
        try:
            gt_candidates = self._explicit_candidates(self._gt, image, lookup)
        except ValueError:
            # Ambiguous GT ownership cannot safely rule out a prediction.
            gt_candidates = ()
        name = image.with_suffix(".xml").name
        nearby, fallback = _nearby_matches(image.parent, name, lookup)
        # Include missing locations: a pending save could create the prediction.
        possible = set(nearby) | {image.with_suffix(".xml"), fallback}
        if len(gt_candidates) == 1:
            possible.discard(gt_candidates[0])
        return possible

    def _possible_prediction_owners(self, path):
        """Identify flat, mirrored and adjacent owners from the XML location."""
        owners = set()
        for candidate in {path, path.resolve()}:
            stem = candidate.stem.casefold()
            parent = candidate.parent
            if self._prediction:
                for directory in {self._prediction.directory, self._prediction.directory.resolve()}:
                    if parent == directory:
                        owners.update(self._by_stem.get(stem, ()))
                    elif self.image_root and parent.is_relative_to(directory):
                        relative_parent = self.image_root / parent.relative_to(directory)
                        owners.update(self._by_location.get((relative_parent, stem), ()))
            else:
                owners.update(self._by_location.get((parent, stem), ()))
                if parent.name.casefold() == "xml":
                    owners.update(self._by_location.get((parent.parent, stem), ()))
        return owners

    def _prediction_paths(self, images, lookup):
        for image in images:
            yield from self._prediction_candidates(image, lookup)

    def first_protected_target(self, paths):
        """Return the first ordinary GT destination that would write to a prediction."""
        names, file_ids = set(), set()
        for pred in self._prediction_paths(self._images, _scan_lookup()):
            resolved = pred.resolve()
            if resolved not in names:
                names.add(resolved)
                identity = _file_id(resolved)
                if identity is not None:
                    file_ids.add(identity)
        for target in paths:
            target = Path(target)
            resolved = target.resolve()
            if resolved in names or _file_id(resolved) in file_ids:
                return target
        return None

    def protects_prediction(self, path):
        """Check a pending save against predictions for the entire incoming dataset."""
        path = Path(path).absolute()
        if self._prediction and self._prediction.bound_file and same_xml_path(path, self._prediction.bound_file):
            return True
        lookup = _scan_lookup()
        candidates = self._possible_prediction_owners(path)

        def matches(images):
            return any(same_xml_path(path, pred) for pred in self._prediction_paths(images, lookup))

        if matches(candidates):
            return True
        # A prediction may be a symlink (even a dangling one) or a differently
        # named hard link to the pending GT. Only a source switch with dirty
        # edits requires this exhaustive alias check.
        return matches(self._images.difference(candidates))

    def _checked_image(self, image):
        image = Path(image).resolve()
        if image not in self._images:
            raise ValueError("图片不在当前 XML 对照列表中。")
        stem = image.stem.casefold()
        if len(self._by_location[image.parent, stem]) > 1 or (not self.image_root and len(self._by_stem[stem]) > 1):
            raise ValueError(f"{image.stem} 有多张同名图片，无法确定对应 XML。")
        return image

    def _resolve(self, image, lookup):
        image = self._checked_image(image)
        name = image.with_suffix(".xml").name
        gt = self._explicit_path(self._gt, image, lookup) if self._gt else None
        prediction = self._explicit_path(self._prediction, image, lookup) if self._prediction else None
        if gt is None:
            gt = _one_match(_nearby_candidates(image, name, prediction, lookup))
        if prediction is None:
            prediction = _one_match(_nearby_candidates(image, name, gt, lookup))
        if same_xml_path(gt, prediction):
            raise ValueError("可编辑标签与只读对照标签指向同一个 XML。请选择不同的文件或目录。")
        self._check_file_binding(image, (gt, prediction))
        # Overlapping source trees and custom file names can make one image's
        # GT point to another image's prediction, even with identical stems.
        for other in self._possible_prediction_owners(gt) - {image}:
            if self._prediction:
                candidates = self._explicit_candidates(self._prediction, other, lookup, protect=True)
            else:
                adjacent = other.with_suffix(".xml")
                nearby, fallback = _nearby_matches(other.parent, adjacent.name, lookup)
                candidates = [*nearby, adjacent, fallback]
            if any(same_xml_path(gt, p) for p in candidates):
                raise ValueError(f"{gt.name} 与另一张图片的只读对照标签冲突。")
        return XmlPair(gt, prediction)
