import glob
import os
import typing as t
from http import HTTPStatus

import requests


class DataSynchronizer:
    def __init__(self, url: str, port: int, user: str, password: str):
        if url[-1] == '/':
            url = url[:-1]
        self.url = url
        self.port = port
        self.user = user
        self.password = password

    def is_exist(self, src, timeout=10):
        try:
            if src.endswith('/'):  # remove the ending '/' if there is
                src = src[:-1]
            r = requests.head(
                f'{self.url}:{self.port}/{src}',
                auth=(self.user, self.password),
                timeout=timeout,
            )
            if r.status_code == HTTPStatus.NOT_FOUND:
                return False
            if r.status_code != HTTPStatus.OK:
                return False
            return True
        except Exception as e:
            print(e)
            return False

    def _put(self, src: str, des: str, timeout=1000000):
        with open(src, 'rb') as f:
            data = f.read()
        r = requests.put(
            f'{self.url}:{self.port}/{des}',
            data=data,
            auth=(self.user, self.password),
            timeout=timeout,
        )
        return r.status_code == HTTPStatus.CREATED

    def _get(self, src: str, des: str, timeout=1000000):
        r = requests.get(
            f'{self.url}:{self.port}/{src}',
            auth=(self.user, self.password),
            timeout=timeout,
            stream=True,
        )
        with open(des, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

        return r.status_code == HTTPStatus.OK

    def create_folder(self, des: str, timeout=1000000):
        _ = requests.request(
            'MKCOL',
            f'{self.url}:{self.port}/{des}',
            auth=(self.user, self.password),
            timeout=timeout,
            stream=True,
        )

    def upload(self, src: str, des: str):
        return self._put(src, des)

    def download(self, src: str, des: str, create_parent_folder=True):
        if create_parent_folder:
            parent_dir = os.path.dirname(des)
            os.makedirs(parent_dir, exist_ok=True)
        return self._get(src, des)

    def upload_folder(self, src_folder: str, des_folder: str, verbose: bool = False):
        if not self.is_folder(src_folder):
            raise ValueError(f'{src_folder} does not end with /')
        if not self.is_folder(des_folder):
            raise ValueError(f'{src_folder} does not end with /')

        file_paths = glob.glob(os.path.join(src_folder, '**/*'), recursive=True)
        for file_path in file_paths:
            if os.path.isdir(file_path):
                continue
            sub_file_path = file_path.replace(src_folder, '', 1)
            des_file_path = os.path.join(des_folder, sub_file_path)
            is_success = self.upload(file_path, des_file_path)
            if verbose:
                status = 'SUCCEED'
                if not is_success:
                    status = 'FAIL'
                print(f'[{status:<7}] {file_path} -> {des_file_path}')

    def download_folder(self, src_folder: str, des_folder: str, verbose: bool = False):
        if not self.is_folder(src_folder):
            raise ValueError(f'{src_folder} does not end with /')
        if not self.is_folder(des_folder):
            raise ValueError(f'{src_folder} does not end with /')

        file_paths = self.parse_file_in_folder(src_folder)

        for file_path in file_paths:
            sub_file_path = file_path.replace(src_folder, '', 1)
            des_file_path = os.path.join(des_folder, sub_file_path)
            is_success = self.download(file_path, des_file_path)
            if verbose:
                status = 'SUCCEED'
                if not is_success:
                    status = 'FAIL'
                print(f'[{status:<7}] {file_path} -> {des_file_path}')

    def parse_file_in_folder(self, des_folder: str) -> t.List[str]:
        r = requests.get(
            f'{self.url}:{self.port}/{des_folder}?json',
            auth=(self.user, self.password),
        )
        path_collection = []
        paths = r.json()['paths']
        for path in paths:
            if path['path_type'] == 'Dir':
                path_collection += self.parse_file_in_folder(os.path.join(des_folder, path['name']))
            else:
                path_collection.append(os.path.join(des_folder, path['name']))
        return path_collection

    def is_folder(self, path: str):
        return path[-1] == '/'
