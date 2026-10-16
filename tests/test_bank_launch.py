import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import manage


class BankLaunchTests(unittest.TestCase):
    def test_database_is_relative_to_workspace_and_env_takes_precedence(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve()
            (root/'config.json').write_text(json.dumps({'bank':{'database':'data/unified.sqlite3'}}))
            with patch.object(manage,'ROOT',root),patch.dict('os.environ',{},clear=True):
                self.assertEqual(manage.bank_database(),str(root/'data/unified.sqlite3'))
                with patch.dict('os.environ',{'BANK_DATABASE':'custom.sqlite3'}):
                    self.assertEqual(manage.bank_database(),str(root/'custom.sqlite3'))

    def test_other_checkout_forwards_bank_to_canonical_python_and_arguments(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve()/'other';root.mkdir()
            target=Path(temp).resolve()/'canonical';(target/'.venv/bin').mkdir(parents=True)
            (target/'.venv/bin/python').touch();(target/'manage.py').touch()
            (root/'config.json').write_text(json.dumps({'bank':{'workspace':str(target)}}))
            with patch.object(manage,'ROOT',root),patch.object(manage.sys,'argv',['manage.py','bank']),patch.object(manage.os,'chdir') as cd,patch.object(manage.os,'execv') as execute:
                manage.forward_bank_workspace()
            cd.assert_called_once_with(target)
            execute.assert_called_once_with(str(target/'.venv/bin/python'),[str(target/'.venv/bin/python'),str(target/'manage.py'),'bank'])

    def test_chained_workspace_redirect_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve()/'a';root.mkdir();target=Path(temp).resolve()/'b';target.mkdir()
            (root/'config.json').write_text(json.dumps({'bank':{'workspace':str(target)}}))
            (target/'config.json').write_text(json.dumps({'bank':{'workspace':str(root)}}))
            with patch.object(manage,'ROOT',root),patch.object(manage.sys,'argv',['manage.py','bank']),patch.object(manage.os,'execv') as execute:
                with self.assertRaises(ValueError):manage.forward_bank_workspace()
            execute.assert_not_called()
