from pathlib import Path

import pandas as pd
from llama_index.core.bridge.pydantic import Field
from llama_index.core.readers.base import BasePydanticReader
from utils import (
    convert_nan_nat_to_none,
    convert_timestamps_to_iso,
    create_uuid_from_string,
    get_utc_time,
    load_pandas,
    num_tokens_from_string,
)

from rag import MIMIC_NODE
from rag.schema import Document, NodeRelationship


class MIMICIIINoteReader(BasePydanticReader):
    """Loads MIMIC-III notes as a list of Document objects."""

    metadata: bool = Field(default=True, description="If metadata to be included or not.")

    include_prev_next_rel: bool = Field(
        default=True, description="Include prev/next node relationships."
    )

    def load_data(
        self,
        file_path: Path | str,
        metadata: bool | None = None,
        extra_info: dict | None = None,
    ) -> list[Document]:
        """Loads list of documents from a MIMIC-III NOTEEVENTS table."""
        return self.load(
            file_path,
            metadata=self.metadata if metadata is None else metadata,
            extra_info=extra_info,
        )

    def load(
        self,
        file_path: Path | str,
        metadata: bool | None = None,
        extra_info: dict | None = None,
    ) -> list[Document]:
        """Loads list of documents from a MIMIC-III NOTEEVENTS table.

        Args:
            file_path (Union[Path, str]): file path of NOTEEVENTS table. The table is
                expected to adopt to the MIMIC-III v1.4 NOTEEVENTS table schema
                (https://mimic.mit.edu/docs/iii/tables/noteevents/)
            metadata (bool, optional): if metadata to be included or not. Defaults to None.
            extra_info (Optional[Dict], optional): extra information related to each document in
                dict format. This gets merged with metadata. Defaults to None.

        Raises:
            TypeError: if file_path is not a string or Path.

        Returns:
            List[Document]: list of Document objects
        """

        metadata = self.metadata if metadata is None else metadata
        extra_info = {} if extra_info is None else extra_info

        # check if file_path is a string or Path
        if not isinstance(file_path, str) and not isinstance(file_path, Path):
            raise TypeError("file_path must be a string or Path.")
        # if extra_info is not None, check if it is a dictionary
        if extra_info and not isinstance(extra_info, dict):
            raise TypeError("extra_info must be a dictionary.")

        # open PDF file
        file_path = Path(file_path)
        noteevents_df = load_pandas(file_path)
        return self.build_nodes_from_pandas(
            noteevents_df, file_path=file_path, metadata=metadata, extra_info=extra_info
        )

    def build_nodes_from_pandas(self, noteevents_df: pd.DataFrame, **kwargs) -> list[Document]:
        """Builds Document objects from a pandas dataframe of notes."""
        file_path = kwargs.get("file_path")
        file_name = file_path.stem if file_path else None
        file_extension = file_path.suffix if file_path else None

        # Sort notes by subject id & date time
        # NOTE: we don't sort by hospital admission because some notes do not have HADM_ID
        noteevents_df = noteevents_df.sort_values(
            ["SUBJECT_ID", "CHARTDATE", "CHARTTIME", "STORETIME"]
        )
        # Group notes by hospital admission, then generate nodes for each group
        nodes = []
        for hadm_id, hadm_df in noteevents_df.groupby("HADM_ID"):
            hadm_nodes = []
            # Generate nodes for each note in the hospital admission
            for i, row in enumerate(hadm_df.itertuples()):
                note_text = row.TEXT
                tokens_ct = num_tokens_from_string(note_text)
                char_ct = len(note_text)
                start_char_idx = 0
                end_char_idx = char_ct

                if kwargs.get("metadata") is True:
                    metadata = {
                        # Monotonically increasing index (relative position in clinical documents)
                        "node_monotonic_idx": i,
                        # Text & File Metadata
                        "num_tokens": tokens_ct,
                        "num_characters": char_ct,
                        "start_char_idx": start_char_idx,
                        "end_char_idx": end_char_idx,
                        "current_page": None,
                        "total_pages": None,
                        "file_path": file_path.as_posix(),
                        "file_name": file_name,
                        "file_extension": file_extension,
                        "node_kind": MIMIC_NODE,
                        "created_at": get_utc_time(),
                        # NOTEEVENT Metadata
                        "ROW_ID": row.ROW_ID,
                        "SUBJECT_ID": row.SUBJECT_ID,
                        "HADM_ID": int(hadm_id),
                        "CHARTDATE": row.CHARTDATE,
                        "CHARTTIME": row.CHARTTIME,
                        "STORETIME": row.STORETIME,
                        "CATEGORY": row.CATEGORY,
                        "DESCRIPTION": row.DESCRIPTION,
                        "CGID": row.CGID,
                    }
                else:
                    metadata = {}
                if extra_info := kwargs.get("extra_info"):
                    metadata = extra_info | metadata

                # Create Unique Node ID by concatenating row values and hashing
                row_str = "|||".join(f"{x}" for x in row)
                node_id = str(create_uuid_from_string(row_str))

                # Make Document object
                node = Document(
                    doc_id=node_id,
                    text=note_text,
                    metadata=metadata,
                    start_char_idx=start_char_idx,
                    end_char_idx=end_char_idx,
                )
                node.metadata = {k: convert_nan_nat_to_none(v) for k, v in node.metadata.items()}
                node.metadata = {k: convert_timestamps_to_iso(v) for k, v in node.metadata.items()}
                hadm_nodes += [node]
            # Add PREVIOUS/NEXT relationships for notes within a single hospital admission
            for i, node in enumerate(hadm_nodes):
                if i > 0:
                    node.relationships[NodeRelationship.PREVIOUS] = hadm_nodes[
                        i - 1
                    ].as_related_node_info()
                if i < len(hadm_nodes) - 1:
                    node.relationships[NodeRelationship.NEXT] = hadm_nodes[
                        i + 1
                    ].as_related_node_info()
            # Combine nodes from all hospital admissions
            nodes.extend(hadm_nodes)
        return nodes
